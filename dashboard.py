"""
Trading dashboard — local web app, this project only.

One page, auto-refreshing, that answers: How is my (paper) account doing?
What is the bot holding? What did it do recently? And what did the research
say? Reads the Alpaca paper account live plus results/summary_*.json.

Run:
    pip install flask
    python dashboard.py           # then open http://127.0.0.1:8050

Works without Alpaca keys too (shows research results only).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

from alpaca_data import load_env
import os

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

app = Flask(__name__)


@app.before_request
def _require_auth():
    """Optional HTTP basic auth, ON only when DASHBOARD_USER/DASHBOARD_PASS
    are set (i.e. when deployed publicly). Local runs leave them unset and
    see no prompt. Protects the account view and the manual-sell endpoint."""
    load_env()
    user = os.environ.get("DASHBOARD_USER")
    pw = os.environ.get("DASHBOARD_PASS")
    if not user or not pw:
        return  # no credentials configured -> local mode, no auth
    from flask import request, Response
    auth = request.authorization
    if not auth or auth.username != user or auth.password != pw:
        return Response("Authentication required", 401,
                        {"WWW-Authenticate": 'Basic realm="Trading dashboard"'})


def _trading_client():
    load_env()
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")
    if not key or not sec:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(key, sec, paper=True)


def _exit_signals():
    """Compute sell signals for open positions: SMA cross-down check and
    distance to trailing stop. Returns list of dicts with exit risk data."""
    try:
        cli = _trading_client()
        if not cli:
            return []
        pos_dict = {p.symbol: p for p in cli.get_all_positions()}
        if not pos_dict:
            return []
        from bot import Bot, Config
        bot = Bot(Config.from_env())
        bars = bot.daily_bars(list(pos_dict.keys()))
        entered = {s: v.get("entered", "") for s, v in
                   bot.state.d.get("positions", {}).items()}

        signals = []
        for sym, p in pos_dict.items():
            if sym not in bars:
                continue
            df = bars[sym]
            c, fast, slow = bot._smas(df)

            # SMA cross-down exit check
            cross_down = (fast.iloc[-1] < slow.iloc[-1] and
                         fast.iloc[-2] >= slow.iloc[-2])

            # Peak SINCE ENTRY — the broker's trailing stop only tracks from
            # when it was created, not historical highs before the buy.
            entry = float(p.avg_entry_price)
            current = float(p.current_price or df["close"].iloc[-1])
            since = df["close"]
            ent_ts = entered.get(sym)
            if ent_ts:
                since = df["close"][df.index >= pd.Timestamp(ent_ts)]
            peak_since_entry = max(entry, current,
                                   float(since.max()) if len(since) else 0)
            stop_level = peak_since_entry * 0.92   # 8% below the peak
            drop_pct = ((peak_since_entry - current) / peak_since_entry) * 100
            distance_to_stop = 8.0 - drop_pct  # pct-points left before stop fires

            signals.append({
                "symbol": sym,
                "current_price": float(round(current, 2)),
                "entry_price": float(round(entry, 2)),
                "peak": float(round(peak_since_entry, 2)),
                "stop_level": float(round(stop_level, 2)),
                "distance_pct": float(round(distance_to_stop, 1)),
                "sma_cross_down": bool(cross_down),
                "qty": int(float(p.qty)),
                "unrealized_pl": float(round(float(p.unrealized_pl), 2)),
            })
        return sorted(signals, key=lambda x: x["distance_pct"])
    except Exception:
        import traceback
        traceback.print_exc()
        return []


def _account_state() -> dict:
    cli = _trading_client()
    if cli is None:
        return {"connected": False, "reason": "no API keys in .env"}
    try:
        acct = cli.get_account()
        positions = [{
            "symbol": p.symbol, "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "price": float(p.current_price or 0),
            "value": float(p.market_value or 0),
            "pl_usd": float(p.unrealized_pl or 0),
            "pl_pct": round(float(p.unrealized_plpc or 0) * 100, 2),
        } for p in cli.get_all_positions()]

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        closed = cli.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, limit=500))
        orders = [{
            "symbol": o.symbol, "side": str(o.side.value),
            "qty": float(o.filled_qty or 0),
            "price": float(o.filled_avg_price or 0),
            "status": str(o.status.value),
            "at": o.filled_at.isoformat() if o.filled_at else
                  (o.submitted_at.isoformat() if o.submitted_at else ""),
        } for o in closed[:25]]
        fills = [{
            "symbol": o.symbol, "side": str(o.side.value),
            "qty": float(o.filled_qty), "price": float(o.filled_avg_price),
            "at": o.filled_at.isoformat(),
            "type": str(getattr(o, "order_type", None) or getattr(o, "type", "")),
            "coid": str(getattr(o, "client_order_id", "") or ""),
        } for o in closed
            if str(o.status.value) == "filled" and o.filled_qty
            and float(o.filled_qty) > 0 and o.filled_avg_price and o.filled_at]

        history = None
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            h = cli.get_portfolio_history(GetPortfolioHistoryRequest(
                period="3M", timeframe="1D"))
            history = {"t": list(h.timestamp), "v": [float(x) if x else None
                                                     for x in h.equity]}
        except Exception:
            pass

        return {
            "connected": True,
            "equity": float(acct.equity or 0),
            "last_equity": float(acct.last_equity or 0),
            "cash": float(acct.cash or 0),
            "buying_power": float(acct.buying_power or 0),
            "positions": positions,
            "orders": orders,
            "journal": build_journal(fills),
            "history": history,
            "kill_switch": (ROOT / "KILL_SWITCH").exists(),
        }
    except Exception as exc:
        return {"connected": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_journal(fills: list[dict]) -> list[dict]:
    """Pair buy fills with the subsequent sell fill per symbol (the bot never
    pyramids: one buy opens a lot, one sell closes it) into completed
    round-trip rows, newest first. Pure function — unit-testable."""
    rows = []
    open_lots: dict[str, dict] = {}
    for f in sorted(fills, key=lambda x: x["at"]):
        if f["side"] == "buy":
            open_lots[f["symbol"]] = f
        elif f["side"] == "sell" and f["symbol"] in open_lots:
            b = open_lots.pop(f["symbol"])
            qty = min(b["qty"], f["qty"])
            pnl = qty * (f["price"] - b["price"])
            t = f["type"].lower()
            reason = ("manual sell (dashboard)"
                      if f.get("coid", "").startswith("manual-")
                      else "8% trailing stop" if "trailing" in t
                      else "SMA cross-down (bot)" if "market" in t
                      else f["type"] or "unknown")
            hold_days = round(
                (pd.Timestamp(f["at"]) - pd.Timestamp(b["at"])).total_seconds()
                / 86400, 1)
            rows.append({
                "symbol": f["symbol"], "strategy": "momentum",
                "entry_at": b["at"], "entry_price": round(b["price"], 4),
                "exit_at": f["at"], "exit_price": round(f["price"], 4),
                "qty": round(qty, 4),
                "pnl_usd": round(pnl, 2),
                "pnl_pct": round((f["price"] / b["price"] - 1) * 100, 2),
                "hold_days": hold_days,
                "reason": reason,
            })
    return rows[::-1]


def _research() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("summary_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        for run, m in d.get("runs", {}).items():
            out.append({
                "file": p.stem.replace("summary_", ""),
                "run": run,
                "window": cfg.get("window", ""),
                "cost_bps": cfg.get("cost_bps_per_side"),
                "trail_pct": round((cfg.get("trail_pct") or 0) * 100, 1),
                "ret": m.get("window_return_pct"),
                "cagr": m.get("cagr_pct"),
                "pf": m.get("profit_factor"),
                "maxdd": m.get("max_drawdown_pct"),
                "sharpe": m.get("sharpe_daily_ann"),
                "trades": m.get("trades"),
                "win": m.get("win_rate_pct"),
            })
        b = d.get("benchmark_spy", {})
        out.append({"file": p.stem.replace("summary_", ""), "run": "SPY b&h",
                    "window": cfg.get("window", ""), "cost_bps": 0,
                    "trail_pct": None, "ret": b.get("window_return_pct"),
                    "cagr": b.get("cagr_pct"), "pf": None,
                    "maxdd": b.get("max_drawdown_pct"),
                    "sharpe": b.get("sharpe_daily_ann"), "trades": 0,
                    "win": None})
    return out


def _json_file(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


@app.route("/api/state")
def api_state():
    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(),
        "account": _account_state(),
        "exit_signals": _exit_signals(),
        "research": _research(),
        "heartbeat": _json_file(ROOT / "heartbeat.json"),
        "alerts": (_json_file(ROOT / "ALERTS.json") or [])[:10],
    })


@app.route("/api/sell/<symbol>", methods=["POST"])
def api_sell(symbol: str):
    """Manual sell from the dashboard: cancel the symbol's protective orders,
    submit a market sell for the full position, sync bot state. PAPER ONLY —
    the trading client is hardcoded paper=True."""
    symbol = symbol.upper().strip()
    cli = _trading_client()
    if cli is None:
        return jsonify({"ok": False, "error": "no API keys"}), 400
    try:
        pos = {p.symbol: p for p in cli.get_all_positions()}.get(symbol)
        if pos is None:
            return jsonify({"ok": False,
                            "error": f"no open position in {symbol}"}), 404

        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
        # cancel THIS symbol's open orders (the trailing stop) first, else the
        # shares are locked and the sell would be rejected
        for o in cli.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[symbol])):
            cli.cancel_order_by_id(o.id)

        order = cli.submit_order(MarketOrderRequest(
            symbol=symbol, qty=float(pos.qty), side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"manual-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{symbol}"))

        # sync bot state so its records match the broker
        try:
            from bot import State
            st = State()
            st.closed(symbol)
        except Exception:
            pass

        return jsonify({"ok": True, "symbol": symbol,
                        "qty": float(pos.qty),
                        "order_status": str(order.status.value)})
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}), 500


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trading dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--tx:#e8eaf0;--mut:#8b93a7;
--up:#22c07a;--dn:#e5484d;--acc:#4f8ef7}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:14px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;padding:24px}
h1{font-size:18px;font-weight:600}
h2{font-size:13px;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin:0 0 12px}
.top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:20px}
.badge{font-size:12px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);color:var(--mut)}
.badge.live{color:var(--up);border-color:var(--up)}
.badge.halt{color:var(--dn);border-color:var(--dn)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.metric .l{font-size:12px;color:var(--mut)}
.metric .v{font-size:24px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.metric .s{font-size:12px;margin-top:2px}
.up{color:var(--up)}.dn{color:var(--dn)}
.wide{margin-bottom:20px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--line);font-size:13px}
tr:last-child td{border-bottom:none}
#empty{color:var(--mut);padding:12px 0}
.foot{color:var(--mut);font-size:12px;margin-top:16px}
td.risk,span.risk{color:#e74c3c;font-weight:bold}
tr.risk{background-color:rgba(231,76,60,0.08)}
tr.warn{background-color:rgba(241,196,15,0.08)}
.num{font-family:monospace}
.sellbtn{background:transparent;color:var(--dn);border:1px solid var(--dn);border-radius:6px;padding:3px 12px;font-size:12px;cursor:pointer}
.sellbtn:hover{background:var(--dn);color:#fff}
.sellbtn:disabled{opacity:.4;cursor:wait}
</style></head><body>
<div class="top">
  <h1>Trading dashboard <span style="color:var(--mut);font-weight:400">&mdash; paper account</span></h1>
  <div id="badges"></div>
</div>

<div id="alerts"></div>
<div class="grid" id="metrics"></div>

<div class="card wide"><h2>Account equity &mdash; last 3 months</h2>
  <div style="height:220px"><canvas id="eq"></canvas></div></div>

<div class="card wide"><h2>Sell signals &mdash; exit risk</h2><div id="exits"></div></div>
<div class="card wide"><h2>Open positions</h2><div id="positions"></div></div>
<div class="card wide"><h2>Trade journal &mdash; completed round trips</h2><div id="journal"></div></div>
<div class="card wide"><h2>Recent fills</h2><div id="orders"></div></div>
<div class="card wide"><h2>Research scoreboard (backtests)</h2><div id="research"></div></div>
<p class="foot" id="foot"></p>

<script>
let chart=null;
const $=id=>document.getElementById(id);
const usd=v=>'$'+Number(v).toLocaleString(undefined,{maximumFractionDigits:0});
const pct=v=>(v>0?'+':'')+Number(v).toFixed(2)+'%';
const cls=v=>v>=0?'up':'dn';

async function refresh(){
  const r=await fetch('/api/state');const d=await r.json();
  const a=d.account;
  let badges='';
  if(d.heartbeat&&d.heartbeat.at){
    const age=(new Date(d.now)-new Date(d.heartbeat.at))/1000;
    if(age<900)badges+='<span class="badge live">bot alive '+(age<90?Math.round(age)+'s':Math.round(age/60)+'m')+' ago'+(d.heartbeat.dry_run?' (dry run)':'')+'</span> ';
    else badges+='<span class="badge halt">bot silent '+Math.round(age/60)+' min — is it running?</span> ';
  } else badges+='<span class="badge">bot not started yet</span> ';
  if(a.connected){
    badges+='<span class="badge live">connected</span> ';
    badges+=a.kill_switch?'<span class="badge halt">KILL SWITCH ON</span>':'<span class="badge">kill switch off</span>';
  } else badges+='<span class="badge halt">not connected: '+(a.reason||'')+'</span>';
  $('badges').innerHTML=badges;

  if(d.alerts&&d.alerts.length){
    let ah='<div class="card wide" style="border-color:var(--dn)"><h2 style="color:var(--dn)">Alerts</h2><table><tr><th>Time</th><th>Severity</th><th>What</th><th>Detail</th></tr>';
    for(const x of d.alerts)ah+='<tr><td>'+x.at.replace('T',' ').slice(0,16)+'</td><td class="'+(x.severity==='critical'?'dn':'')+'">'+x.severity+'</td><td>'+x.title+'</td><td style="text-align:left">'+(x.detail||'')+'</td></tr>';
    $('alerts').innerHTML=ah+'</table></div>';
  } else $('alerts').innerHTML='';

  if(a.connected){
    const day=a.equity-a.last_equity, dayp=a.last_equity?day/a.last_equity*100:0;
    $('metrics').innerHTML=
      metric('Account value',usd(a.equity),pct(dayp)+' today',cls(day))+
      metric('Today’s change',usd(day),'vs yesterday’s close',cls(day))+
      metric('Cash available',usd(a.cash),'buying power '+usd(a.buying_power),'')+
      metric('Margin cushion',usd(a.buying_power-a.equity),'PDT rule retired Jun 2026','');
    exitSignalsTable(d.exit_signals);
    positionsTable(a.positions);
    journalTable(a.journal);
    ordersTable(a.orders);
    if(a.history&&a.history.v){drawChart(a.history)}
  } else {
    $('metrics').innerHTML=metric('Account','—','connect Alpaca keys in .env','');
    $('exits').innerHTML='<div id="empty">Not connected.</div>';
    $('positions').innerHTML='<div id="empty">Not connected.</div>';
    $('journal').innerHTML='<div id="empty">Not connected.</div>';
    $('orders').innerHTML='<div id="empty">Not connected.</div>';
  }
  researchTable(d.research);
  $('foot').textContent='Refreshed '+new Date().toLocaleTimeString()+' · auto-refreshes every 60s · paper trading only — no real money';
}
function metric(l,v,s,c){return '<div class="card metric"><div class="l">'+l+'</div><div class="v '+c+'">'+v+'</div><div class="s '+c+'">'+s+'</div></div>'}
function exitSignalsTable(es){
  if(!es||!es.length){$('exits').innerHTML='<div id="empty">No open positions.</div>';return}
  let h='<table><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th><th>Peak</th><th>Stop level (8%)</th><th>Distance to stop</th><th>Unrealized</th><th>SMA exit?</th></tr>';
  for(const x of es){
    let risk='';
    if(x.distance_pct<2)risk='risk';
    else if(x.distance_pct<4)risk='warn';
    h+='<tr class="'+risk+'"><td><strong>'+x.symbol+'</strong></td><td>'+x.qty+'</td><td class="num">'+x.entry_price.toFixed(2)+'</td><td class="num">'+x.current_price.toFixed(2)+'</td><td class="num">'+x.peak.toFixed(2)+'</td><td class="num">'+x.stop_level.toFixed(2)+'</td><td class="num '+(x.distance_pct<2?'risk':'')+'">'+x.distance_pct.toFixed(1)+' pts</td><td class="'+cls(x.unrealized_pl)+'">'+usd(x.unrealized_pl)+'</td><td>'+(x.sma_cross_down?'<span class="risk">YES</span>':'—')+'</td></tr>';
  }
  $('exits').innerHTML=h+'</table><p style="font-size:0.85em;margin-top:0.5em">Distance to stop: how many percentage points the price can still fall before the 8% trailing stop sells (8 = at peak, 0 = selling now). Red &lt;2 pts: stop imminent. Yellow &lt;4 pts: watch.</p>';
}
function positionsTable(ps){
  if(!ps||!ps.length){$('positions').innerHTML='<div id="empty">No open positions.</div>';return}
  let h='<table><tr><th>Symbol</th><th>Qty</th><th>Avg entry</th><th>Now</th><th>Value</th><th>P&amp;L $</th><th>P&amp;L %</th><th></th></tr>';
  for(const p of ps)h+='<tr><td>'+p.symbol+'</td><td>'+p.qty+'</td><td>'+p.avg_entry.toFixed(2)+'</td><td>'+p.price.toFixed(2)+'</td><td>'+usd(p.value)+'</td><td class="'+cls(p.pl_usd)+'">'+usd(p.pl_usd)+'</td><td class="'+cls(p.pl_pct)+'">'+pct(p.pl_pct)+'</td><td><button class="sellbtn" onclick="manualSell(\''+p.symbol+'\','+p.qty+','+p.pl_usd.toFixed(2)+')">Sell</button></td></tr>';
  $('positions').innerHTML=h+'</table>';
}
async function manualSell(sym,qty,pl){
  if(!confirm('Sell ALL '+qty+' shares of '+sym+' at market price now?\n\nCurrent P&L: $'+pl+'\n\nThis cancels its trailing stop and submits a market sell to Alpaca (paper account).'))return;
  const btns=document.querySelectorAll('.sellbtn');btns.forEach(b=>b.disabled=true);
  try{
    const r=await fetch('/api/sell/'+sym,{method:'POST'});
    const d=await r.json();
    if(d.ok){alert(sym+' sell order submitted ('+d.qty+' shares, status: '+d.order_status+'). Refreshing...');}
    else{alert('Sell FAILED: '+d.error);}
  }catch(e){alert('Sell request failed: '+e);}
  refresh();
}
function journalTable(js){
  if(!js||!js.length){$('journal').innerHTML='<div id="empty">No completed trades yet — rows appear when a position is bought AND sold.</div>';return}
  const t=s=>s?s.replace('T',' ').slice(0,16):'';
  let h='<table><tr><th>Symbol</th><th>Bought</th><th>Buy price</th><th>Sold</th><th>Sell price</th><th>Qty</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Held</th><th>Strategy</th><th>Exit via</th></tr>';
  for(const x of js)h+='<tr><td>'+x.symbol+'</td><td>'+t(x.entry_at)+'</td><td>'+x.entry_price.toFixed(2)+'</td><td>'+t(x.exit_at)+'</td><td>'+x.exit_price.toFixed(2)+'</td><td>'+x.qty+'</td><td class="'+cls(x.pnl_usd)+'">'+usd(x.pnl_usd)+'</td><td class="'+cls(x.pnl_pct)+'">'+pct(x.pnl_pct)+'</td><td>'+x.hold_days+'d</td><td>'+x.strategy+'</td><td>'+x.reason+'</td></tr>';
  $('journal').innerHTML=h+'</table>';
}
function ordersTable(os){
  if(!os||!os.length){$('orders').innerHTML='<div id="empty">No recent fills.</div>';return}
  let h='<table><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Fill price</th><th>Status</th></tr>';
  for(const o of os)h+='<tr><td>'+(o.at?o.at.replace('T',' ').slice(0,16):'')+'</td><td>'+o.symbol+'</td><td class="'+(o.side==='buy'?'up':'dn')+'">'+o.side+'</td><td>'+o.qty+'</td><td>'+(o.price?o.price.toFixed(2):'—')+'</td><td>'+o.status+'</td></tr>';
  $('orders').innerHTML=h+'</table>';
}
function researchTable(rs){
  if(!rs||!rs.length){$('research').innerHTML='<div id="empty">No backtest results yet.</div>';return}
  let h='<table><tr><th>Backtest</th><th>Strategy</th><th>Window</th><th>Return</th><th>CAGR</th><th>PF</th><th>Max DD</th><th>Sharpe</th><th>Trades</th></tr>';
  for(const x of rs)h+='<tr><td>'+x.file+'</td><td>'+x.run+'</td><td>'+x.window+'</td><td class="'+cls(x.ret||0)+'">'+(x.ret==null?'—':pct(x.ret))+'</td><td>'+(x.cagr==null?'—':pct(x.cagr))+'</td><td>'+(x.pf==null?'—':x.pf)+'</td><td class="dn">'+(x.maxdd==null?'—':x.maxdd+'%')+'</td><td>'+(x.sharpe==null?'—':x.sharpe)+'</td><td>'+(x.trades||0)+'</td></tr>';
  $('research').innerHTML=h+'</table>';
}
function drawChart(hist){
  const pts=hist.t.map((t,i)=>({x:new Date(t*1000).toLocaleDateString(),y:hist.v[i]})).filter(p=>p.y!=null);
  if(chart)chart.destroy();
  chart=new Chart($('eq'),{type:'line',data:{labels:pts.map(p=>p.x),datasets:[{data:pts.map(p=>p.y),borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:.2,fill:false}]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
  scales:{x:{ticks:{color:'#8b93a7',maxTicksLimit:8},grid:{display:false}},
  y:{ticks:{color:'#8b93a7',callback:v=>'$'+Number(v).toLocaleString()},grid:{color:'#262b36'}}}}});
}
refresh();setInterval(refresh,60000);
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    # PORT/HOST from env for cloud hosts (Railway sets PORT); localhost default.
    port = int(os.environ.get("PORT", "8050"))
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
