# fpl/ai_manager/core.py
import streamlit as st
import pandas as pd
from fpl.api import fetch_fixtures, fetch_player_history

SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_SHAPE = {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}
MAX_PER_CLUB = 3


def _avg_fdr_next(team_id: int, n: int = 3) -> float:
    fdf = pd.DataFrame(fetch_fixtures())
    sub = fdf[
        ((fdf["team_h"] == team_id) | (fdf["team_a"] == team_id))
        & (fdf["finished"] == False)
    ]
    if sub.empty:
        return 3.0
    sub = sub.sort_values("kickoff_time").head(n)
    fdrs = [
        (g["team_h_difficulty"] if g["team_h"] == team_id else g["team_a_difficulty"])
        for _, g in sub.iterrows()
    ]
    return float(sum(fdrs) / len(fdrs)) if fdrs else 3.0


def build_scored_market(players_df: pd.DataFrame, n_fixt: int = 3) -> pd.DataFrame:
    team_fdr = {}
    for tid in players_df["team"].unique():
        team_fdr[int(tid)] = _avg_fdr_next(int(tid), n=n_fixt)

    def _row_score(r):
        import pandas as pd

        form = float(pd.to_numeric(r.get("form", 0), errors="coerce") or 0)
        ppg = float(pd.to_numeric(r.get("points_per_game", 0), errors="coerce") or 0)
        mins = float(pd.to_numeric(r.get("minutes", 0), errors="coerce") or 0)
        fdr = float(team_fdr.get(int(r["team"]), 3.0))
        status_pen = 0.0 if str(r.get("status", "a")).lower() in ("a", "d") else -2.0
        playtime = min(1.0, mins / 1800.0)
        return (
            ((form * 0.6 + ppg * 0.8) + status_pen)
            * (6.0 - fdr)
            * (0.6 + 0.4 * playtime)
        )

    market = players_df.copy()
    market["score"] = market.apply(_row_score, axis=1)
    return market


def pick_starting_xi(scored_squad: pd.DataFrame):
    by_pos = {
        p: scored_squad[scored_squad["pos"] == p]
        .sort_values("score", ascending=False)
        .copy()
        for p in ["GK", "DEF", "MID", "FWD"]
    }
    xi_ids = []
    xi_ids += by_pos["GK"].head(1)["id"].tolist()
    xi_ids += by_pos["DEF"].head(XI_SHAPE["DEF"])["id"].tolist()
    xi_ids += by_pos["MID"].head(XI_SHAPE["MID"])["id"].tolist()
    xi_ids += by_pos["FWD"].head(XI_SHAPE["FWD"])["id"].tolist()
    xi_df = scored_squad[scored_squad["id"].isin(xi_ids)].copy()
    bench_pool = scored_squad[~scored_squad["id"].isin(xi_ids)].copy()
    gk2 = bench_pool[bench_pool["pos"] == "GK"].sort_values("score", ascending=False)
    outfield_bench = bench_pool[bench_pool["pos"] != "GK"].sort_values(
        "score", ascending=False
    )
    bench_ids = (gk2["id"].tolist()[:1] + outfield_bench["id"].tolist()[:3])[:4]
    cap_row = xi_df.sort_values("score", ascending=False).iloc[0]
    return xi_df, {"captain_id": int(cap_row["id"]), "bench_ids": bench_ids}


def event_points_for(pid: int, gw: int) -> int:
    try:
        h = fetch_player_history(pid)
        for g in h.get("history", []):
            if int(g.get("round")) == int(gw):
                return int(g.get("total_points", 0))
    except Exception:
        pass
    return 0


def simulate_week_points(
    xi_df: pd.DataFrame,
    captain_id: int,
    gw: int,
    chip: str | None = None,
    bench_ids: list[int] | None = None,
) -> int:
    xi_pts = sum(event_points_for(int(pid), gw) for pid in xi_df["id"].tolist())
    cap_pts = event_points_for(int(captain_id), gw)
    total = xi_pts + cap_pts
    if chip == "TC":
        total += cap_pts
    if chip == "BB" and bench_ids:
        total += sum(event_points_for(int(pid), gw) for pid in bench_ids)
    return int(total)


def ensure_auto_state(players_df: pd.DataFrame, _teams_df_unused):
    if "auto_mgr" in st.session_state:
        return
    # ownership-seeded template (~100m)
    budget, club_ct, squad = 100.0, {}, []
    for pos, need in SQUAD_SHAPE.items():
        cand = (
            players_df[players_df["pos"] == pos]
            .sort_values("selected_by", ascending=False)
            .copy()
        )
        for _, r in cand.iterrows():
            if len([1 for s in squad if s["pos"] == pos]) >= need:
                break
            club = r["team_short"]
            if club_ct.get(club, 0) >= MAX_PER_CLUB:
                continue
            price = float(r["price"])
            if budget - price < 0:
                continue
            squad.append({"id": int(r["id"]), "buy_price": price, "pos": pos})
            club_ct[club] = club_ct.get(club, 0) + 1
            budget -= price
    st.session_state.auto_mgr = {
        "squad": [{"id": s["id"], "buy_price": s["buy_price"]} for s in squad],
        "bank": round(budget, 1),
        "free_transfers": 1,
        "last_gw_processed": None,
        "chips": {"TC": True, "BB": True, "FH": True, "WC1": True, "WC2": True},
        "log": [],
    }
