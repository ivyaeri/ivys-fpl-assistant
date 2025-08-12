# ui/tab_fixtures.py
import streamlit as st

def render_fixtures_tab(fixtures_text):
    st.subheader("Upcoming Fixtures by Team")
    for row in fixtures_text:
        st.write(row)
