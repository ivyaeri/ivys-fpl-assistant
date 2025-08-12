# app.py
import os, hashlib
import streamlit as st
import pandas as pd

from config import TZ, MODEL_NAME
from fpl.kb import build_full_kb
from fpl.api import fetch_bootstrap, fetch_fixtures, fetch_player_history
from fpl.ai_manager.core import ensure_auto_state
from fpl.ai_manager.decision import run_ai_auto_until_current
from fpl.ai_manager.persist import (
    load_manager_state, save_manager_state, maybe_git_commit_and_push
)
from ui.tabs_leaderboards import render_top20, render_top10_by_pos, render_budget
from ui.tab_fixtures import render_fixtures_tab
from ui.tab_chat import render_chat_tab
from ui.tab_ai_auto import render_ai_tabs

st.set_page_config(page_title="FPL Chat Agent", page_icon="⚽", layout="wide")
st.title("⚽ FPL Assistant")

# ---- Sidebar (API key + options)
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

with st.sidebar:
    st.subheader("🔐 API & Options")
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        value=st.session_state.openai_key,
        type="password",
        key="openai_api_key_input",
        help="Stored only in your session.",
    )
    if api_key_input:
        st.session_state.openai_key = api_key_input.strip()

    include_hist = st.checkbox("Include recent player history", value=True)
    last_n = st.slider("Recent GWs", 3, 8, 5, 1)

    if st.button("🔄 Refresh live KB"):
        for k in ["full_kb", "kb_meta", "players_df", "fixtures_text", "kb_hash", "conversation"]:
            st.session_state.pop(k, None)

    st.caption(f"API key present: {'Yes' if st.session_state.openai_key else 'No'}")

# ---- Build/refresh KB
if "full_kb" not in st.session_state:
    full_kb, kb_meta, players_df, fixtures_text = build_full_kb(include_hist, last_n)
    st.session_state.full_kb = full_kb
    st.session_state.kb_meta = kb_meta
    st.session_state.players_df = players_df
    st.session_state.fixtures_text = fixtures_text
    st.session_state.kb_hash = hashlib.sha256(full_kb.encode("utf-8")).hexdigest()
else:
    full_kb = st.session_state.full_kb
    kb_meta = st.session_state.kb_meta
    players_df = st.session_state.players_df
    fixtures_text = st.session_state.fixtures_text

st.caption(kb_meta["header"])

# ---- Load persistent AI manager state (or seed template)
persisted = load_manager_state()
if persisted is not None:
    st.session_state.auto_mgr = persisted  # restore full season state
else:
    teams_df_seed = pd.DataFrame(fetch_bootstrap().get("teams", []))
    ensure_auto_state(players_df, teams_df_seed)

# ---- Run AI manager to current GW (idempotent), then persist + maybe git push
run_ai_auto_until_current(
    kb_meta=kb_meta,
    players_df=players_df,
    teams_df=pd.DataFrame(fetch_bootstrap().get("teams", [])),
    fixtures=fetch_fixtures(),
    fetch_player_history=fetch_player_history,
    model_name=MODEL_NAME,
)
save_manager_state(st.session_state.auto_mgr)
maybe_git_commit_and_push()

# ---- Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Top 20 Overall","Top 10 by Position","Top Budget Picks","Fixtures","Chat Agent","Auto Manager","AI Auto Manager"]
)

with tab1:
    render_top20(players_df)
with tab2:
    render_top10_by_pos(players_df)
with tab3:
    render_budget(players_df)
with tab4:
    render_fixtures_tab(st.session_state.fixtures_text)
with tab5:
    render_chat_tab(model_name=MODEL_NAME, kb_text=st.session_state.full_kb, kb_hash=st.session_state.kb_hash)
with tab6:
    render_ai_tabs.render_auto_baseline(players_df, kb_meta)   # quick baseline view
with tab7:
    render_ai_tabs.render_ai_weekly(players_df, kb_meta)       # full weekly AI log view
