# ui/tabs_leaderboards.py
import streamlit as st

def render_top20(players_df):
    st.subheader("Top 20 Players by Ownership")
    st.dataframe(
        players_df.sort_values("selected_by", ascending=False).head(20)[
            ["web_name","team_short","pos","price","form","selected_by"]
        ],
        use_container_width=True
    )

def render_top10_by_pos(players_df):
    st.subheader("Top 10 by Position (Ownership)")
    for pos_name in ["GK","DEF","MID","FWD"]:
        st.markdown(f"**{pos_name}**")
        st.dataframe(
            players_df[players_df["pos"] == pos_name]
            .sort_values("selected_by", ascending=False)
            .head(10)[["web_name","team_short","price","form","selected_by"]],
            use_container_width=True
        )

def render_budget(players_df):
    st.subheader("Top Budget Picks (≤ £5.0m)")
    st.dataframe(
        players_df[players_df["price"] <= 5.0]
        .sort_values("selected_by", ascending=False)
        .head(15)[["web_name","team_short","pos","price","form","selected_by"]],
        use_container_width=True
    )
