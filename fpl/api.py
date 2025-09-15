# fpl/api.py
import requests
from functools import lru_cache

FPL_API = "https://fantasy.premierleague.com/api"
REQ_TIMEOUT = 10  # seconds

@lru_cache(maxsize=8)
def fetch_bootstrap():
    r = requests.get(f"{FPL_API}/bootstrap-static/")
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=8)
def fetch_fixtures():
    r = requests.get(f"{FPL_API}/fixtures/")
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=512)
def fetch_player_history(player_id: int):
    r = requests.get(f"{FPL_API}/element-summary/{player_id}/")
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=512)
def fetch_event_live(gw: int) -> dict:
    """
    Fetch official per-player stats for a specific gameweek (live or finished).
    gw: int - the gameweek number (1-based)
    Returns JSON dict with "elements": list of players.
    """
    url = f"{FPL_API}/event/{gw}/live/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_entry_event(entry_id: int, gw: int) -> dict:
    """Official FPL endpoint: entry picks + entry_history for a given GW."""
    r = requests.get(f"{FPL_API}/entry/{int(entry_id)}/event/{int(gw)}/picks/")
    r.raise_for_status()
    return r.json()

def fetch_entry_history(entry_id: int) -> dict:
    """Overall history (optional, useful if you want totals/overall rank)."""
    r = requests.get(f"{FPL_API}/entry/{int(entry_id)}/history/")
    r.raise_for_status()
    return r.json()

