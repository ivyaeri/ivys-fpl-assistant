# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd

from config import MODEL_NAME
from fpl.ai_manager.decision import (
    ensure_initial_squad_with_ai,
    rewind_and_regenerate_current_gw,
    run_ai_auto_until_current,
    refresh_logged_points,
    force_redraft_gw1,  # NEW: allow full GW1 re-draft on demand
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

    # ---------- No squad yet: offer AI GW1 draft, show preview, then process GW ----------
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
        if gw_now == 1:
            st.markdown("---")
            force_redraft_toggle = st.checkbox(
                "Force full re-draft for **GW1** (replace all 15 via AI, no FT cost)",
                value=False,
                help="Uses the drafter again; applies your note and may revise the current 15.",
            )
        else:
            force_redraft_toggle = False

    colA, colB = st.columns([1, 3])
    with colA:
        regen_disabled = not bool(st.session_state.openai_key)
        if st.button("🔁 Regenerate this GW (AI)", type="primary", disabled=regen_disabled):
            with st.spinner("Re-evaluating this gameweek…"):
                # If GW1 and user wants a full redraft, do it first, then log the week
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
                    extra_instructions=(user_note or None),  # note applies for THIS regenerate only
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
        if entry.get("redraft"):
            header.append("Full redraft")

        with st.expander(" — ".join(header), expanded=(entry["gw"] == gw_now)):
            # Transfers (support both single-transfer and multi-move schemas)
            if entry.get("redraft"):
                st.markdown("**Full redraft applied.**")
            elif entry.get("moves"):
                if len(entry["moves"]) == 0:
                    st.markdown("**No transfer made.**")
                else:
                    for mv in entry["moves"]:
                        st.markdown(f"**Transfer:** {_pname(players_df, mv['out'])} → {_pname(players_df, mv['in'])}")
            elif entry.get("made") and entry.get("transfer"):
                out_id = entry["transfer"]["out"]
                in_id = entry["transfer"]["in"]
                st.markdown(f"**Transfer:** {_pname(players_df, out_id)} → {_pname(players_df, in_id)}")
            else:
                st.markdown("**No transfer made.**")

            st.markdown(f"**Reason (AI):** {entry.get('reason', '')}")

            # Prepare lists/sets
            xi_list = list(map(int, entry.get("xi_ids", [])))
            bench_list = list(map(int, entry.get("bench_ids") or entry.get("bench_order") or []))
            cap_id = int(entry.get("captain_id") or 0)
            xi_ids = set(xi_list)
            bench_ids = set(bench_list)
            squad_ids = list(map(int, entry.get("squad_ids", [])))

            week = players_df[players_df["id"].isin(squad_ids)].copy()
            if not week.empty:
                week["XI"] = week["id"].apply(lambda x: "Yes" if x in xi_ids else "")
                week["Bench"] = week["id"].apply(lambda x: "Yes" if x in bench_ids else "")
                week["Captain"] = week["id"].apply(lambda x: "C" if x == cap_id else "")
                week = week[
                    [
                        "web_name",
                        "team_short",
                        "pos",
                        "price",
                        "form",
                        "status",
                        "selected_by",
                        "points_per_game",
                        "XI",
                        "Bench",
                        "Captain",
                    ]
                ].sort_values(["Captain", "XI", "pos", "web_name"], ascending=[False, False, True, True])

                st.markdown("**Full 15-man squad (this GW):**")
                st.dataframe(week, use_container_width=True)
                st.markdown(f"**Captain:** {_pname(players_df, cap_id) if cap_id else '—'}")
            else:
                st.info("Squad snapshot not available for this entry.")
