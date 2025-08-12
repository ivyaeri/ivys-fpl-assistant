# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd
from fpl.ai_manager.core import build_scored_market, pick_starting_xi

class render_ai_tabs:
    @staticmethod
    def render_auto_baseline(players_df, kb_meta):
        st.subheader("🤖 Auto Manager — Current Squad & XI (baseline view)")
        if "auto_mgr" not in st.session_state:
            st.info("Auto manager not initialized.")
            return
        state = st.session_state.auto_mgr
        market = build_scored_market(players_df, n_fixt=3)
        squad_ids = [s["id"] for s in state["squad"]]
        in_squad = players_df[players_df["id"].isin(squad_ids)][
            ["web_name","team_short","pos","price","form","selected_by","status"]
        ].sort_values(["pos","web_name"])
        st.markdown("**Current 15-man Squad**")
        st.dataframe(in_squad, use_container_width=True)

        xi_df, meta = pick_starting_xi(market[market["id"].isin(squad_ids)])
        cap_name = xi_df.loc[xi_df["id"] == meta["captain_id"], "web_name"].iloc[0] if not xi_df.empty else "—"
        st.markdown(f"**Current XI (3-4-3)** — Captain: **{cap_name}**")
        st.dataframe(xi_df[["web_name","team_short","pos","price","form","score"]].sort_values(["pos","web_name"]), use_container_width=True)

        st.markdown("### Weekly Log (summary)")
        if not state["log"]:
            st.info("No gameweeks processed yet.")
        else:
            for entry in sorted(state["log"], key=lambda x: x["gw"], reverse=True):
                with st.expander(f"GW {entry['gw']} — Points: {entry['points']} — Bank: £{entry['bank']:.1f}m — FTs: {entry['free_transfers']}"):
                    if entry["made"] and entry["transfer"]:
                        st.markdown(f"**Transfer:** {entry['transfer']['out']} → {entry['transfer']['in']}")
                    else:
                        st.markdown("**No transfer made.**")
                    if entry.get("chip") and entry["chip"] != "NONE":
                        st.markdown(f"**Chip used:** {entry['chip']}")
                    st.markdown(f"**Reason:** {entry.get('reason','')}")

    @staticmethod
    def render_ai_weekly(players_df, kb_meta):
        st.subheader("🧠 AI Auto Manager — Weekly Decisions & Full Squad")
        if "auto_mgr" not in st.session_state:
            st.info("Auto manager not initialized.")
            return
        state = st.session_state.auto_mgr

        if not state["log"]:
            st.info("No gameweeks processed yet.")
            return

        def pname(pid: int) -> str:
            row = players_df.loc[players_df["id"] == pid]
            return row["web_name"].iloc[0] if not row.empty else f"ID {pid}"

        for entry in sorted(state["log"], key=lambda x: x["gw"], reverse=True):
            header = [f"GW {entry['gw']}", f"Points: {entry['points']}", f"Bank £{entry['bank']:.1f}", f"FTs {entry['free_transfers']}"]
            if entry.get("chip") and entry["chip"] != "NONE": header.append(f"Chip {entry['chip']}")
            with st.expander(" — ".join(header), expanded=(entry["gw"] == kb_meta.get("gw"))):
                if entry["made"] and entry["transfer"]:
                    st.markdown(f"**Transfer:** {pname(entry['transfer']['out'])} → {pname(entry['transfer']['in'])}")
                else:
                    st.markdown("**No transfer made.**")
                st.markdown(f"**Reason (AI):** {entry.get('reason','')}")

                xi_ids = set(entry.get("xi_ids", []))
                bench_ids = set(entry.get("bench_ids", []))
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
                st.markdown(f"**Captain:** {pname(cap_id) if cap_id else '—'}")
