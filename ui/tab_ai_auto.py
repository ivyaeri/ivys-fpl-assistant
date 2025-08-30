# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd

from config import MODEL_NAME
from fpl.ai_manager.decision import (
    ensure_initial_squad_with_ai,
    rewind_and_regenerate_current_gw,
    run_ai_auto_until_current,
    refresh_logged_points,
    force_redraft_gw1,
)

# ---------- helpers ----------
def _maps(players_df: pd.DataFrame):
    """Return (code->id, id->code). If columns are missing, return empties."""
    if "code" in players_df.columns and "id" in players_df.columns:
        c2i = {int(c): int(i) for c, i in zip(players_df["code"], players_df["id"])}
        i2c = {int(i): int(c) for i, c in zip(players_df["id"], players_df["code"])}
        return c2i, i2c
    return {}, {}

def _pname_by_code(players_df: pd.DataFrame, code: int) -> str:
    if "code" not in players_df.columns:
        return f"CODE {code}"
    row = players_df.loc[players_df["code"] == int(code)]
    return row["web_name"].iloc[0] if not row.empty else f"CODE {code}"

def _pname_by_id(players_df: pd.DataFrame, pid: int) -> str:
    if "id" not in players_df.columns:
        return f"ID {pid}"
    row = players_df.loc[players_df["id"] == int(pid)]
    return row["web_name"].iloc[0] if not row.empty else f"ID {pid}"

def _snapshot_df(players_df: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    """Small table for the UI based on codes."""
    if "code" not in players_df.columns:
        return pd.DataFrame()
    sub = players_df[players_df["code"].isin(list(map(int, codes)))].copy()
    if sub.empty:
        return sub
    cols = [
        "web_name","team_short","pos","price","form","status","selected_by","points_per_game","code"
    ]
    return sub[[c for c in cols if c in sub.columns]]

# ---------- main ----------
def render_ai_tab(players_df: pd.DataFrame, kb_meta: dict, user_id: str):
    st.subheader("🧠 AI Auto Manager — LLM-only")

    if "auto_mgr" not in st.session_state:
        st.info("State not loaded yet.")
        return

    state = st.session_state.auto_mgr
    c2i, i2c = _maps(players_df)

    # ---------- No squad yet: offer AI GW1 draft, then process ----------
    if not state.get("squad"):
        col1, col2 = st.columns([1, 3])

        with col1:
            disabled = not bool(st.session_state.openai_key)
            if st.button("🧠 Draft GW1 Squad (AI)", disabled=disabled):
                with st.spinner("Asking the model to draft your 15..."):
                    ensure_initial_squad_with_ai(
                        user_id=user_id,
                        players_df=players_df,
                        kb_text=st.session_state.full_kb,
                        model_name=MODEL_NAME,
                        budget=100.0,
                    )

                squad_codes = st.session_state.get("auto_mgr", {}).get("squad") or []
                if len(squad_codes) == 15:
                    with st.spinner("Locking in GW decisions…"):
                        run_ai_auto_until_current(
                            user_id=user_id,
                            kb_meta=kb_meta,
                            players_df=players_df,
                            model_name=MODEL_NAME,
                            extra_instructions=None,
                        )
                    st.success("Drafted and processed the current GW. See the log below.")
                    st.rerun()
                else:
                    reason = st.session_state.get("auto_mgr", {}).get("seed_origin", "unknown")
                    st.error(f"Draft failed ({reason}). Check your API key and try again.")

        with col2:
            if not st.session_state.openai_key:
                st.info("Add your OpenAI API key in the sidebar, then click **Draft GW1 Squad (AI)**.")
            else:
                st.info("The model will pick a legal 15 (2 GK / 5 DEF / 5 MID / 3 FWD, ≤3/club, ≤£100m).")
        return

    # ---------- With a squad: controls & regenerate ----------
    gw_now = int(kb_meta.get("gw") or 0)

    with st.expander("Optional: add instructions / redraft controls", expanded=False):
        st.caption("Examples: “prefer Arsenal defenders”, “avoid flagged players”, “consider BB if bench is strong”.")
        user_note = st.text_area(
            "Manager instructions (optional)",
            value="",
            height=90,
            placeholder="Type any constraints or preferences…",
        )
        # simple auto-redraft toggle for GW1 (always on if GW1)
        force_redraft_toggle = (gw_now == 1)

    colA, colB = st.columns([1, 3])
    with colA:
        regen_disabled = not bool(st.session_state.openai_key)
        if st.button("🔁 Regenerate this GW (AI)", type="primary", disabled=regen_disabled):
            with st.spinner("Re-evaluating this gameweek…"):
                if gw_now == 1 and force_redraft_toggle:
                    ok, msg = force_redraft_gw1(
                        user_id=user_id,
                        players_df=players_df,
                        kb_text=st.session_state.full_kb,
                        model_name=MODEL_NAME,
                        extra_instructions=(user_note or None),
                    )
                    if not ok:
                        st.error(f"Redraft failed: {msg}")
                        st.stop()

                ok, msg = rewind_and_regenerate_current_gw(
                    user_id=user_id,
                    kb_meta=kb_meta,
                    players_df=players_df,
                    model_name=MODEL_NAME,
                    extra_instructions=(user_note or None),
                )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)
    with colB:
        st.caption(f"User: **{user_id}**  ·  Current GW: **{gw_now or '—'}**")

    # ---------- Maintenance ----------
    with st.expander("Maintenance", expanded=False):
        st.caption("Recompute points for all logged GWs from official FPL history (useful after a GW finishes).")
        if st.button("↻ Refresh points for finished GWs"):
            n = refresh_logged_points(user_id, players_df=players_df, kb_meta=kb_meta)
            st.success(f"Updated {n} gameweek(s).")
            st.rerun()

    # ---------- Weekly logs ----------
    logs = state.get("log") or []
    if not logs:
        st.info("No gameweeks processed yet.")
        return

    for entry in sorted(logs, key=lambda x: x["gw"], reverse=True):
        header = [
            f"GW {entry['gw']}",
            f"Points: {entry.get('points', 0)}",
            f"Bank £{float(entry.get('bank', 0.0)):.1f}",
            f"FTs {int(entry.get('free_transfers', 0))}",
        ]
        chip = (entry.get("chip") or "NONE").upper()
        if chip != "NONE":
            header.append(f"Chip {chip}")
        if entry.get("redraft"):
            header.append("Full redraft")

        with st.expander(" — ".join(header), expanded=(entry["gw"] == gw_now)):
            # --- Transfers: prefer code fields, fallback to legacy
            moves = entry.get("transfers") or entry.get("moves") or []
            if isinstance(moves, dict):  # very old single-transfer schema
                moves = [moves]

            if not moves:
                st.markdown("**No transfer made.**")
            else:
                for mv in moves:
                    out_code = mv.get("out_code")
                    in_code  = mv.get("in_code")
                    if out_code is None and "out" in mv:
                        # legacy id or ambiguous field; try map
                        out_id = mv["out"]
                        out_code = i2c.get(int(out_id), out_id)
                    if in_code is None and "in" in mv:
                        in_id = mv["in"]
                        in_code = i2c.get(int(in_id), in_id)
                    st.markdown(f"**Transfer:** {_pname_by_code(players_df, out_code)} → {_pname_by_code(players_df, in_code)}")

            # Reasons
            treasons = entry.get("transfer_reasons") or []
            treasons_txt = ("; ".join(treasons)) if isinstance(treasons, list) else treasons
            st.markdown(f"**Reason (AI):** {entry.get('reason','')}")
            st.markdown(f"**Bench Reason (AI):** {entry.get('bench_reason','')}")
            if treasons_txt:
                st.caption(f"Transfer notes: {treasons_txt}")

            # --- XI/Bench/Squad (prefer codes; fallback via mapping)
            xi_codes = entry.get("xi_codes")
            if not xi_codes and entry.get("xi_ids"):
                xi_codes = [int(i2c.get(int(x), x)) for x in entry["xi_ids"]]
            bench_codes = entry.get("bench_codes")
            if not bench_codes:
                legacy_bench = entry.get("bench_ids") or entry.get("bench_order") or []
                bench_codes = [int(i2c.get(int(x), x)) for x in legacy_bench]
            cap_code = entry.get("captain_code")
            if cap_code is None and entry.get("captain_id") is not None:
                cap_code = int(i2c.get(int(entry["captain_id"]), entry["captain_id"]))
            squad_codes = entry.get("squad_codes")
            if not squad_codes and entry.get("squad_ids"):
                squad_codes = [int(i2c.get(int(x), x)) for x in entry["squad_ids"]]

            # Build squad table
            week = _snapshot_df(players_df, squad_codes or [])
            if not week.empty:
                xi_set = set(map(int, xi_codes or []))
                bench_set = set(map(int, bench_codes or []))
                cap_code = int(cap_code or 0)

                week["XI"] = week["code"].apply(lambda c: "Yes" if int(c) in xi_set else "")
                week["Bench"] = week["code"].apply(lambda c: "Yes" if int(c) in bench_set else "")
                week["Captain"] = week["code"].apply(lambda c: "C" if int(c) == cap_code else "")

                week = week[
                    [
                        "web_name","team_short","pos","price","form","status","selected_by","points_per_game",
                        "XI","Bench","Captain"
                    ]
                ].sort_values(["Captain", "XI", "pos", "web_name"], ascending=[False, False, True, True])

                st.markdown("**Full 15-man squad (this GW):**")
                st.dataframe(week, use_container_width=True)
                st.markdown(f"**Captain:** {_pname_by_code(players_df, cap_code) if cap_code else '—'}")
            else:
                st.info("Squad snapshot not available for this entry.")
