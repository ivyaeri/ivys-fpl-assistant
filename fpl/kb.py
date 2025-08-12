# fpl/kb.py
from datetime import datetime
import pandas as pd
import pytz

from fpl.api import fetch_bootstrap, fetch_fixtures, fetch_player_history

TZ = pytz.timezone("Europe/London")
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def _recent_block(pid: int, last_n: int = 5):
    try:
        h = fetch_player_history(pid)
        hist = h.get("history", [])[-last_n:]
        if not hist: return "RECENT: n/a"
        pts = [int(g.get("total_points", 0)) for g in hist]
        mins = [int(g.get("minutes", 0)) for g in hist]
        goals = sum(int(g.get("goals_scored", 0)) for g in hist)
        assists = sum(int(g.get("assists", 0)) for g in hist)
        cs = sum(int(g.get("clean_sheets", 0)) for g in hist)
        return f"RECENT({len(pts)}): pts[{','.join(map(str,pts))}] | avg {sum(pts)/len(pts):.2f} | mins/90 {sum(mins)/90.0:.1f} | G{goals} A{assists} CS{cs}"
    except Exception:
        return "RECENT: n/a"

def build_full_kb(include_history: bool = True, last_n: int = 5):
    bs = fetch_bootstrap()
    fixtures = fetch_fixtures()
    events = pd.DataFrame(bs.get("events", []))
    players = pd.DataFrame(bs.get("elements", []))
    teams = pd.DataFrame(bs.get("teams", []))

    team_short = teams.set_index("id")["short_name"].to_dict()

    players = players.copy()
    players["team_short"] = players["team"].map(team_short)
    players["price"] = players["now_cost"] / 10.0
    players["pos"] = players["element_type"].map(POS)
    players["selected_by"] = pd.to_numeric(players.get("selected_by_percent", 0), errors="coerce").fillna(0.0)

    cols = ["id","web_name","team_short","pos","price","form","selected_by","status","news","minutes","points_per_game","total_points","ict_index"]
    keep = [c for c in cols if c in players.columns]
    p_lines = []
    for _, r in players[keep].iterrows():
        pid = int(r.get("id"))
        base = (
            f"PLAYER: {r.get('web_name')} | TEAM: {r.get('team_short')} | POS: {r.get('pos')} | "
            f"PRICE: £{float(r.get('price', 0)):.1f}m | FORM: {r.get('form')} | OWN: {float(r.get('selected_by', 0)):.1f}% | "
            f"PPG: {r.get('points_per_game')} | TOT: {r.get('total_points')} | MINS: {r.get('minutes')} | "
            f"ICT: {r.get('ict_index')} | STATUS: {r.get('status')} | NEWS: {str(r.get('news') or '')[:120]}"
        )
        if include_history:
            base += " | " + _recent_block(pid, last_n=last_n)
        p_lines.append(base)

    fx = pd.DataFrame(fixtures)
    fx = fx[fx["finished"] == False].copy()
    team_fx_lines = []
    if not fx.empty:
        for tid in sorted(teams["id"].tolist()):
            sub = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)].sort_values("kickoff_time").head(last_n)
            if sub.empty: continue
            parts = []
            for _, g in sub.iterrows():
                is_home = g["team_h"] == tid
                opp = g["team_a"] if is_home else g["team_h"]
                fdr = g["team_h_difficulty"] if is_home else g["team_a_difficulty"]
                opps = team_short.get(int(opp), str(opp))
                gw = g.get("event")
                parts.append(f"GW{gw} {'vs' if is_home else '@'} {opps} (FDR {fdr})")
            team_fx_lines.append(f"TEAM_FIX: {team_short.get(tid, str(tid))} → " + "; ".join(parts))

    gw_now = None
    if not events.empty:
        if "is_current" in events.columns and events["is_current"].any():
            gw_now = int(events.loc[events["is_current"] == True, "id"].iloc[0])
        else:
            upcoming = events[events["finished"] == False].sort_values("deadline_time")
            if not upcoming.empty:
                gw_now = int(upcoming["id"].iloc[0])

    header = f"KB_BUILT: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} | CURRENT_GW: {gw_now} | PLAYERS: {len(p_lines)}"
    full_kb = f"{header}\n\n[FIXTURES]\n" + "\n".join(team_fx_lines) + "\n\n[PLAYERS]\n" + "\n".join(p_lines)
    return full_kb, {"gw": gw_now, "players": len(p_lines), "header": header}, players, team_fx_lines
