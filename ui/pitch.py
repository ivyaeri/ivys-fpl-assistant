# ui/pitch.py
import streamlit as st
import pandas as pd
from typing import Iterable

# Minimal, responsive FPL-style pitch using HTML/CSS (works in Streamlit)
_PITCH_CSS = """
<style>
.pitch-wrap { margin: 8px 0 16px 0; }
.pitch {
  position: relative;
  border-radius: 16px;
  padding: 22px 10px;
  background:
    repeating-linear-gradient(
      0deg,
      #1e7d32 0px, #1e7d32 44px,
      #237a34 44px, #237a34 88px
    );
  box-shadow: inset 0 0 0 2px rgba(255,255,255,0.15);
}
.pitch .row {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 10px;
  margin: 10px 0;
  flex-wrap: wrap;
}
.player {
  min-width: 90px;
  max-width: 120px;
  padding: 8px 8px 10px 8px;
  border-radius: 12px;
  text-align: center;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.18);
  color: #fff;
  position: relative;
  backdrop-filter: blur(2px);
}
.player .name { font-weight: 600; font-size: 0.92rem; line-height: 1.1; }
.player .meta { font-size: 0.78rem; opacity: 0.9; }
.player .pos  { font-size: 0.72rem; opacity: 0.85; letter-spacing: .02em; }
.badge-c {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 22px; height: 22px;
  border-radius: 50%;
  background: #ffd60a; color: #111;
  font-weight: 800; font-size: .78rem;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid rgba(0,0,0,0.2);
}
.bench {
  margin-top: 10px;
  padding: 10px;
  border-radius: 12px;
  background: rgba(0,0,0,0.06);
  border: 1px dashed rgba(255,255,255,0.2);
}
.bench-title {
  color: rgba(255,255,255,0.9);
  font-size: .9rem;
  margin: 0 0 6px 2px;
}
@media (max-width: 420px) {
  .player { min-width: 86px; }
}
</style>
"""

def _player_card(row: pd.Series, is_captain: bool = False) -> str:
    name = row.get("web_name", f"ID {int(row['id'])}")
    team = row.get("team_short", "")
    price = f"£{float(row.get('price', 0.0)):.1f}m"
    pos = row.get("pos", "")
    badge = '<div class="badge-c">C</div>' if is_captain else ""
    return f"""
    <div class="player">
      {badge}
      <div class="name">{name}</div>
      <div class="meta">{team} · {price}</div>
      <div class="pos">{pos}</div>
    </div>
    """

def render_pitch(players_df: pd.DataFrame, xi_ids: Iterable[int], bench_ids: Iterable[int], captain_id: int | None):
    """
    Render an FPL-style pitch: GK row, DEF row, MID row, FWD row; bench below.
    - players_df must have: id, web_name, team_short, pos, price
    - xi_ids: 11 ids; bench_ids: 4 ids in order; captain_id in xi_ids
    """
    xi_ids = [int(x) for x in (xi_ids or [])]
    bench_ids = [int(x) for x in (bench_ids or [])]
    cap = int(captain_id or 0)

    if len(xi_ids) == 0:
        st.info("No XI available to render.")
        return

    xi = players_df[players_df["id"].isin(xi_ids)].copy()
    bench = players_df[players_df["id"].isin(bench_ids)].copy()

    # Group XI by position
    gk  = xi[xi["pos"] == "GK"].sort_values("web_name")
    defs = xi[xi["pos"] == "DEF"].sort_values("web_name")
    mids = xi[xi["pos"] == "MID"].sort_values("web_name")
    fwds = xi[xi["pos"] == "FWD"].sort_values("web_name")

    # Inject CSS once
    st.markdown(_PITCH_CSS, unsafe_allow_html=True)

    # Build HTML rows
    def _row_html(df: pd.DataFrame) -> str:
        cards = "".join(_player_card(r, is_captain=(int(r["id"]) == cap)) for _, r in df.iterrows())
        return f'<div class="row">{cards}</div>'

    pitch_html = f"""
    <div class="pitch-wrap">
      <div class="pitch">
        {_row_html(gk)}
        {_row_html(defs)}
        {_row_html(mids)}
        {_row_html(fwds)}
        <div class="bench">
          <div class="bench-title">Bench</div>
          {_row_html(bench)}
        </div>
      </div>
    </div>
    """

    st.markdown(pitch_html, unsafe_allow_html=True)
