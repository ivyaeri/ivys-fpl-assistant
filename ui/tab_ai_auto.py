# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd

from config import MODEL_NAME
from fpl.ai_manager.decision import (
    ensure_initial_squad_with_ai,
    rewind_and_regenerate_current_gw,
    run_ai_auto_until_current,  
    refresh_logged_points,   # NEW: recompute points for finished GWs
)

def _pname(players_df: pd.DataFrame, pid: int) -> str:
    row = players_df.loc[players_df["id"] == pid]
    return row["web_name"].iloc[0] if not row.empty else f"ID {pid}"

def render_ai_tab(players_df: pd.DataFrame, kb_meta: dict, user_id: str):
    st.subheader("🧠 AI Auto Manager — LLM-only")

    if "auto_mgr" not in st.session_state:
        st.info("State not loaded yet.")
        return

    state = st.session_state.auto_mgr

    # ---------- No squad yet: offer AI GW1 draft, show preview, auto-rerun ----------
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
                squad_ids = (st.session_state.get("auto_mgr", {}).get("squad") or [])
                if len(squad_ids) == 15:
                    # ✅ Immediately process the current GW so a row gets logged
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

    # ---------- With a squad: custom instructions + regenerate ----------
    with st.expander("Optional: add instructions for this week’s regenerate", expanded=False):
        st.caption("e.g., “prefer Arsenal defenders”, “avoid flagged players”, “consider a BB if bench has strong fixtures”.")
        user_note = st.text_area(
            "Manager instructions (optional)",
            value="",
            height=90,
            placeholder="Type any constraints or preferences…"
        )
        st.caption("These instructions are passed to the LLM for **this** regenerate only. All FPL rules still apply.")

    colA, colB = st.columns([1, 3])
    with colA:
        regen_disabled = not bool(st.session_state.openai_key)
        if st.button("🔁 Regenerate this GW (AI)", type="primary", disabled=regen_disabled):
            with st.spinner("Re-evaluating this gameweek with your instructions…"):
                ok, msg = rewind_and_regenerate_current_gw(
                    user_id=user_id,
                    kb_meta=kb_meta,
                    players_df=players_df,
                    model_name=MODEL_NAME,
                    extra_instructions=(user_note or None),  # pass the optional note
                )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)
    with colB:
        st.caption(f"User: **{user_id}**")

    # ---------- Maintenance utilities ----------
    with st.expander("Maintenance", expanded=False):
        st.caption("Recompute points for all logged GWs from official FPL history (useful after a GW finishes).")
        if st.button("↻ Refresh points for finished GWs"):
            n = refresh_logged_points(user_id)
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
            f"Points: {entry['points']}",
            f"Bank £{entry['bank']:.1f}",
            f"FTs {entry['free_transfers']}",
        ]
        if entry.get("chip") and entry["chip"] != "NONE":
            header.append(f"Chip {entry['chip']}")

        with st.expander(" — ".join(header), expanded=(entry["gw"] == kb_meta.get("gw"))):
            # Transfer summary
            if entry.get("made") and entry.get("transfer"):
                out_id = entry["transfer"]["out"]
                in_id = entry["transfer"]["in"]
                st.markdown(f"**Transfer:** {_pname(players_df, out_id)} → {_pname(players_df, in_id)}")
            else:
                st.markdown("**No transfer made.**")

            st.markdown(f"**Reason (AI):** {entry.get('reason', '')}")

            xi_ids = set(entry.get("xi_ids", []))
            # support older key name "bench_order"
            bench_ids = set(entry.get("bench_ids", []) or entry.get("bench_order", []))
            cap_id = entry.get("captain_id")
            squad_ids = entry.get("squad_ids", [])

            week = players_df[players_df["id"].isin(squad_ids)].copy()
            if not week.empty:
                week["XI"] = week["id"].apply(lambda x: "Yes" if x in xi_ids else "")
                week["Bench"] = week["id"].apply(lambda x: "Yes" if x in bench_ids else "")
                week["Captain"] = week["id"].apply(lambda x: "C" if x == cap_id else "")
                week = week[
                    ["web_name","team_short","pos","price","form","status","selected_by","points_per_game","XI","Bench","Captain"]
                ].sort_values(["Captain","XI","pos","web_name"], ascending=[False, False, True, True])

                st.markdown("**Full 15-man squad (this GW):**")
                st.dataframe(week, use_container_width=True)
                st.markdown(f"**Captain:** {_pname(players_df, cap_id) if cap_id else '—'}")
            else:
                st.info("Squad snapshot not available for this entry.")
