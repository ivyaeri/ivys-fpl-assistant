# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd
from config import MODEL_NAME, SEASON
from fpl.ai_manager.decision import (
    ensure_initial_squad_with_ai,
    rewind_and_regenerate_current_gw,
)
from fpl.ai_manager.persist_db import get_gw_logs  # display past from DB

def _pname(players_df: pd.DataFrame, pid: int) -> str:
    row = players_df.loc[players_df["id"] == pid]
    return row["web_name"].iloc[0] if not row.empty else f"ID {pid}"

def render_ai_tab(players_df: pd.DataFrame, kb_meta: dict, user_id: str):
    st.subheader("🧠 AI Auto Manager — LLM-only")

    if "auto_mgr" not in st.session_state:
        st.info("State not loaded yet.")
        return
    state = st.session_state.auto_mgr

    if not state.get("squad"):
        col1, col2 = st.columns([1,3])
        with col1:
            disabled = not bool(st.session_state.openai_key)
            if st.button("🧠 Draft GW1 Squad (AI)", disabled=disabled):
                ensure_initial_squad_with_ai(
                    user_id=user_id,
                    players_df=players_df,
                    kb_text=st.session_state.full_kb,
                    model_name=MODEL_NAME,
                    budget=100.0,
                )
                st.success("Drafted (if AI succeeded). Click ▶ Rerun.")
                st.stop()
        with col2:
            st.info("Add your OpenAI API key in the sidebar, then click **Draft GW1 Squad (AI)**.")
        return

    colA, colB = st.columns([1,3])
    with colA:
        disabled = not bool(st.session_state.openai_key)
        if st.button("🔁 Regenerate this GW (AI)", disabled=disabled):
            ok, msg = rewind_and_regenerate_current_gw(
                user_id=user_id,
                kb_meta=kb_meta,
                players_df=players_df,
                model_name=MODEL_NAME,
            )
            st.success(msg if ok else f"Nothing changed: {msg}")

    with colB:
        st.caption(f"User: **{user_id}** · Season: **{SEASON}**")

    # Show latest in-memory + DB history
    logs = (state.get("log") or [])[:]
    db_logs = get_gw_logs(user_id)  # includes earlier GWs (immutable)
    # merge unique by gw
    seen = {int(e["gw"]) for e in logs}
    for e in db_logs:
        if int(e["gw"]) not in seen:
            logs.append(e)

    if not logs:
        st.info("No gameweeks processed yet.")
        return

    for entry in sorted(logs, key=lambda x: x["gw"], reverse=True):
        header = [f"GW {entry['gw']}", f"Points: {entry['points']}", f"Bank £{entry['bank']:.1f}", f"FTs {entry['free_transfers']}"]
        if entry.get("chip") and entry["chip"] != "NONE": header.append(f"Chip {entry['chip']}")
        with st.expander(" — ".join(header), expanded=(entry["gw"] == kb_meta.get("gw"))):
            if entry["made"] and entry["transfer"]:
                st.markdown(f"**Transfer:** {_pname(players_df, entry['transfer']['out'])} → {_pname(players_df, entry['transfer']['in'])}")
            else:
                st.markdown("**No transfer made.**")
            st.markdown(f"**Reason (AI):** {entry.get('reason','')}")

            xi_ids = set(entry.get("xi_ids", []))
            bench_ids = entry.get("bench_ids", [])
            cap_id = entry.get("captain_id")
            squad_ids = entry.get("squad_ids", [])

            week = players_df[players_df["id"].isin(squad_ids)].copy()
            week["XI"] = week["id"].apply(lambda x: "Yes" if x in xi_ids else "")
            week["Bench"] = week["id"].apply(lambda x: "Yes" if x in bench_ids else "")
            week["Captain"] = week["id"].apply(lambda x: "C" if x == cap_id else "")
            week = week[["web_name","team_short","pos","price","form","status","selected_by","points_per_game","XI","Bench","Captain"]]
            week = week.sort_values(["Captain","XI","pos","web_name"], ascending=[False, False, True, True])
            st.markdown("**Full 15-man squad (this GW):**")
            st.dataframe(week, use_container_width=True)
            st.markdown(f"**Captain:** {_pname(players_df, cap_id) if cap_id else '—'}")
