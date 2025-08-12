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
