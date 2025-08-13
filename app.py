# app.py
import os, hashlib
import streamlit as st
import pandas as pd

from fpl.kb import build_full_kb
from fpl.api import fetch_bootstrap
if "DATABASE_URL" in st.secrets:
    os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
from fpl.ai_manager.persist_db import init_db, load_state, save_state
from fpl.ai_manager.decision import ensure_initial_squad_with_ai, run_ai_auto_until_current
from ui.tabs_leaderboards import render_top20, render_top10_by_pos, render_budget
from ui.tab_fixtures import render_fixtures_tab
from ui.tab_chat import render_chat_tab
from ui.tab_ai_auto import render_ai_tab

from config import TZ, MODEL_NAME

st.set_page_config(page_title="FPL Chat Agent", page_icon="⚽", layout="wide")
st.title("⚽ FPL Assistant")

# DB init
init_db()
# NOTE: this wrapper will cache the KB indefinitely; we’ll bust it manually.

@st.cache_data(show_spinner=False)
def get_full_kb_cached(epoch: int):
    """
    Cache the KB until the user explicitly refreshes.
    The sole cache key is `epoch`, which we bump from the Refresh button.
    We read the current toggle/slider values from session_state.
    """
    include_history = st.session_state.get("include_hist", False)
    last_n = st.session_state.get("last_n", 5)
    # uses your existing build_full_kb(...) defined below in this file
    return build_full_kb(include_history=include_history, last_n=last_n)
# Sidebar
if "openai_key" not in st.session_state: st.session_state.openai_key = ""
if "user_id" not in st.session_state:    st.session_state.user_id = "default"

with st.sidebar:
    st.subheader("👤 User")
    # text input writes to a temp key so we can control when to apply it
    uid_val = st.text_input("User ID", value=st.session_state.user_id, key="uid_input")

    def _apply_user():
        st.session_state.user_id = (st.session_state.uid_input or "demo").strip()
        # clear per-user in-memory state so the app reloads from DB for the new user
        for k in ["auto_mgr"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.button("Use this ID", on_click=_apply_user, help="Switch active user/profile")

    st.caption(f"Active user: **{st.session_state.user_id}**")
        # now it's safe to call DB with the active user_id
    persisted = load_state(st.session_state.user_id)
    if persisted is not None:
        st.session_state.auto_mgr = persisted
    else:
        st.session_state.auto_mgr = st.session_state.get("auto_mgr", {"squad": []})
        
    st.subheader("🔐 API & Options")

    # your existing API key field
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        value=st.session_state.get("openai_key", ""),
        type="password",
        key="openai_api_key_input",
        help="Stored only in your session.",
    )
    if api_key_input:
        st.session_state.openai_key = api_key_input.strip()

    # ✅ Persist the knobs; they won’t apply until you click Refresh
    include_hist = st.checkbox(
        "Include recent player history",
        value=st.session_state.get("include_hist", False)
    )
    last_n = st.slider(
        "Recent GWs",
        3, 8,
        st.session_state.get("last_n", 5),
        1
    )
    st.session_state.include_hist = include_hist
    st.session_state.last_n = last_n

    # ✅ Epoch to control the manual cache
    if "kb_epoch" not in st.session_state:
        st.session_state.kb_epoch = 0

    # ✅ Refresh button: bump epoch + clear cached func + clear session copies
    if st.button("🔄 Refresh live KB"):
        st.session_state.kb_epoch += 1
        try:
            get_full_kb_cached.clear()
        except Exception:
            pass
        for k in ["full_kb", "kb_meta", "players_df", "fixtures_text", "kb_hash", "conversation"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.caption("KB is cached until you click **Refresh live KB**. Changes above apply on next refresh.")
    st.caption(f"API key present: {'Yes' if st.session_state.get('openai_key') else 'No'}")

# Build KB
# --------------- Build / refresh FULL KB (manual-cache) ---------------
full_kb, kb_meta, players_df, fixtures_text = get_full_kb_cached(st.session_state.kb_epoch)
st.session_state.full_kb = full_kb
st.session_state.kb_meta = kb_meta
st.session_state.players_df = players_df
st.session_state.fixtures_text = fixtures_text
st.session_state.kb_hash = hashlib.sha256(full_kb.encode("utf-8")).hexdigest()

st.caption(kb_meta["header"])

# Load state from DB (don't auto-call the LLM here)
persisted = load_state(st.session_state.user_id)
if persisted is not None:
    st.session_state.auto_mgr = persisted
else:
    # Keep an empty state so the UI can render; GW1 draft happens only when triggered
    st.session_state.auto_mgr = st.session_state.get("auto_mgr", {"squad": []})

# Trigger AI only when requested
trigger_ai = st.sidebar.button("▶ Initialize/Run AI now") or st.session_state.get("auto_kick", False)

if trigger_ai:
    if not st.session_state.openai_key:
        st.sidebar.warning("Add your OpenAI API key first.")
    else:
        with st.spinner("Running AI manager…"):
            ensure_initial_squad_with_ai(
                user_id=st.session_state.user_id,
                players_df=players_df,
                kb_text=st.session_state.full_kb,
                model_name=MODEL_NAME,
                budget=100.0,
            )
            run_ai_auto_until_current(
                user_id=st.session_state.user_id,
                kb_meta=kb_meta,
                players_df=players_df,
                model_name=MODEL_NAME,
            )
        st.sidebar.success("AI manager updated.")
        st.rerun()


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Top 20 Overall","Top 10 by Position","Top Budget Picks","Fixtures","AI Auto Manager","Chat"]
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
    render_ai_tab(players_df, kb_meta, user_id=st.session_state.user_id)
with tab6:
    render_chat_tab(
        model_name=MODEL_NAME,
        kb_text=st.session_state.full_kb,
        kb_hash=st.session_state.kb_hash,
    )


