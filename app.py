# app.py
import os, hashlib
import streamlit as st
import pandas as pd

from fpl.kb import build_full_kb
from fpl.api import fetch_bootstrap
from fpl.ai_manager.persist_db import init_db, load_state, save_state
from fpl.ai_manager.decision import ensure_initial_squad_with_ai, run_ai_auto_until_current
from ui.tabs_leaderboards import render_top20, render_top10_by_pos, render_budget
from ui.tab_fixtures import render_fixtures_tab
from ui.tab_chat import render_chat_tab
from ui.tab_ai_auto import render_ai_tab

st.set_page_config(page_title="FPL Chat Agent", page_icon="⚽", layout="wide")
st.title("⚽ FPL Assistant")

# DB init
init_db()

# Sidebar
if "openai_key" not in st.session_state: st.session_state.openai_key = ""
if "user_id" not in st.session_state:    st.session_state.user_id = "default"

with st.sidebar:
    st.subheader("👤 User")
    user_id = st.text_input("User ID", value=st.session_state.user_id, help="Per-user state in DB.")
    if user_id: st.session_state.user_id = user_id.strip()

    st.subheader("🔐 API & Options")
    api_key_input = st.text_input("Enter your OpenAI API key", value=st.session_state.openai_key, type="password")
    if api_key_input: st.session_state.openai_key = api_key_input.strip()

    include_hist = st.checkbox("Include recent player history", value=True)
    last_n = st.slider("Recent GWs", 3, 8, 5, 1)

    if st.button("🔄 Refresh live KB"):
        for k in ["full_kb","kb_meta","players_df","fixtures_text","kb_hash","conversation"]:
            st.session_state.pop(k, None)
    st.caption(f"API key present: {'Yes' if st.session_state.openai_key else 'No'}")

# Build KB
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

# Load state from DB or let AI draft initial squad (if API present). No greedy fallback.
persisted = load_state(st.session_state.user_id)
if persisted is not None:
    st.session_state.auto_mgr = persisted
else:
    ensure_initial_squad_with_ai(
        user_id=st.session_state.user_id,
        players_df=players_df,
        kb_text=st.session_state.full_kb,
        model_name=MODEL_NAME,
        budget=100.0,
    )

# Run AI to current GW (LLM only). Persist along the way.
run_ai_auto_until_current(
    user_id=st.session_state.user_id,
    kb_meta=kb_meta,
    players_df=players_df,
    model_name=MODEL_NAME,
)

# Tabs
tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(
    ["Top 20 Overall","Top 10 by Position","Top Budget Picks","Fixtures","AI Auto Manager", "Chat Assistant"]
)
with tab1: render_top20(players_df)
with tab2: render_top10_by_pos(players_df)
with tab3: render_budget(players_df)
with tab4: render_fixtures_tab(st.session_state.fixtures_text)
with tab5: render_ai_tab(players_df, kb_meta, user_id=st.session_state.user_id)
with tab6: renedr_chat_tab(model_name, kb_meta,kb_hash)

from config import TZ, MODEL_NAME
