# fpl/kb.py
from datetime import datetime
import pandas as pd
import pytz

from fpl.api import fetch_bootstrap, fetch_fixtures, fetch_player_history, fetch_event_live

TZ = pytz.timezone("Europe/London")
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}



def _fetch_gw_points(gw: int, id_to_code: dict) -> dict[int, int]:
    """Return {player_code: points} for a given GW."""
    try:
        live = fetch_event_live(gw)  # wraps /event/{gw}/live/
        elements = live.get("elements", [])
        pts_map = {}
        for e in elements:
            el_id = int(e["id"])
            code = id_to_code.get(el_id)  # stable player code
            if not code:
                continue
            stats = e.get("stats", {})
            total = int(stats.get("total_points", 0))
            pts_map[int(code)] = total
        return pts_map
    except Exception:
        return {}

def _recent_block(element_id: int, last_n: int = 5):
    """element_id is the season-specific FPL 'id' (NOT code)."""
    try:
        h = fetch_player_history(element_id)
        hist = h.get("history", [])[-last_n:]
        if not hist:
            return "RECENT: n/a"
        pts = [int(g.get("total_points", 0)) for g in hist]
        mins = [int(g.get("minutes", 0)) for g in hist]
        goals = sum(int(g.get("goals_scored", 0)) for g in hist)
        assists = sum(int(g.get("assists", 0)) for g in hist)
        cs = sum(int(g.get("clean_sheets", 0)) for g in hist)
        return (
            f"RECENT({len(pts)}): pts[{','.join(map(str, pts))}] | "
            f"avg {sum(pts)/len(pts):.2f} | mins/90 {sum(mins)/90.0:.1f} | "
            f"G{goals} A{assists} CS{cs}"
        )
    except Exception:
        return "RECENT: n/a"

def build_full_kb(include_history: bool = True, last_n: int = 15):
    bs = fetch_bootstrap()
    fixtures = fetch_fixtures()

    events = pd.DataFrame(bs.get("events", []))
    players = pd.DataFrame(bs.get("elements", []))
    teams = pd.DataFrame(bs.get("teams", []))

    team_short = teams.set_index("id")["short_name"].to_dict()
    STATUS_LABEL = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable"}

    # ---- Enrich players and compute stable mappings ----
    players = players.copy()
    players["team_short"] = players["team"].map(team_short)
    players["price"] = players["now_cost"] / 10.0
    players["pos"] = players["element_type"].map(POS)
    players["selected_by"] = pd.to_numeric(players.get("selected_by_percent", 0), errors="coerce").fillna(0.0)
    players["chance_next"] = pd.to_numeric(players.get("chance_of_playing_next_round"), errors="coerce")
    players["chance_this"] = pd.to_numeric(players.get("chance_of_playing_this_round"), errors="coerce")
    players["status_label"] = players["status"].map(STATUS_LABEL).fillna(players["status"])

    # Stable identity → use FPL 'code'
    # Keep both so we can resolve code -> current season 'id' when needed
    code_to_id = dict(zip(players["code"].astype(int), players["id"].astype(int)))
    id_to_code = dict(zip(players["id"].astype(int), players["code"].astype(int)))
    name_to_code = {str(w).lower(): int(c) for w, c in zip(players["web_name"], players["code"])}

    cols = [
        "code", "id", "web_name", "team_short", "pos", "price", "form", "selected_by",
        "status", "news", "minutes", "points_per_game", "total_points", "ict_index",
        "chance_next", "status_label", "chance_this"
    ]
    keep = [c for c in cols if c in players.columns]

    p_lines = []
    for _, r in players[keep].iterrows():
        code = int(r.get("code"))              # stable across seasons
        current_id = int(r.get("id"))          # volatile each season (used only for history)
        base = (
            f"PLAYER: {r['web_name']} | CODE: {code} | TEAM: {r['team_short']} | POS: {r['pos']} | "
            f"PRICE: £{float(r['price']):.1f}m | FORM: {r['form']} | OWN: {float(r['selected_by']):.1f}% | "
            f"PPG: {r['points_per_game']} | TOT: {r['total_points']} | MINS: {r['minutes']} | ICT: {r['ict_index']} | "
            f"STATUS: {r['status_label']} ({'' if pd.isna(r['chance_next']) else int(r['chance_next'])}% next) | "
            f"NEWS: {str(r.get('news') or '')[:120]}"
        )
        if include_history:
            base += " | " + _recent_block(current_id, last_n=last_n)
        p_lines.append(base)

    # ---- Team fixtures (upcoming) ----
    fx = pd.DataFrame(fixtures)
    fx = fx[fx["finished"] == False].copy()
    team_fx_lines = []
    if not fx.empty:
        for tid in sorted(teams["id"].tolist()):
            sub = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)].sort_values("kickoff_time").head(last_n)
            if sub.empty:
                continue
            parts = []
            for _, g in sub.iterrows():
                is_home = g["team_h"] == tid
                opp = g["team_a"] if is_home else g["team_h"]
                fdr = g["team_h_difficulty"] if is_home else g["team_a_difficulty"]
                opps = team_short.get(int(opp), str(opp))
                gw = g.get("event")
                parts.append(f"GW{gw} {'vs' if is_home else '@'} {opps} (FDR {fdr})")
            team_fx_lines.append(f"TEAM_FIX: {team_short.get(tid, str(tid))} → " + "; ".join(parts))

    # ---- Determine upcoming GW ----
    now_uk = datetime.now(TZ)
    gw_now = None
    if not events.empty and "deadline_time" in events.columns:
        ev = events.copy()
        ev["_dl"] = pd.to_datetime(ev["deadline_time"], utc=True, errors="coerce").dt.tz_convert(TZ)
        ev = ev.dropna(subset=["_dl"]).sort_values("_dl").reset_index(drop=True)

        if not ev.empty:
            if now_uk < ev.loc[0, "_dl"]:
                gw_now = int(ev.loc[0, "id"])
            else:
                after = ev.index[ev["_dl"] > now_uk]
                if len(after) > 0:
                    gw_now = int(ev.loc[after[0], "id"])
                else:
                    gw_now = int(ev.loc[len(ev) - 1, "id"])
    else:
        gw_now = None

    header = f"KB_BUILT: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} | CURRENT_GW: {gw_now} | PLAYERS: {len(p_lines)}"
    full_kb = f"{header}\n\n[FIXTURES]\n" + "\n".join(team_fx_lines) + "\n\n[PLAYERS]\n" + "\n".join(p_lines)
        # ---- Historical GW points (last 5) ----
    points_by_gw = {}
    max_gw = int(events["id"].max()) if not events.empty else 0
    for gw in range(max(1, (gw_now or 1) - 5), (gw_now or 1)):
        pts_map = _fetch_gw_points(gw, id_to_code)
        if pts_map:
            points_by_gw[gw] = pts_map

    meta = {
        "gw": gw_now,
        "players": len(p_lines),
        "header": header,
        "code_to_id": code_to_id,
        "id_to_code": id_to_code,
        "name_to_code": name_to_code,
        "points_by_gw": points_by_gw,   # 👈 now available for refresh
    }

    return full_kb, meta, players, team_fx_lines
