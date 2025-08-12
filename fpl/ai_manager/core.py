# fpl/ai_manager/core.py
# Pure constants & simple helpers; no greedy scoring/selection.
SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3

# Valid FPL starting formations:
VALID_FORMATIONS = {(3,4,3),(3,5,2),(4,3,3),(4,4,2),(4,5,1),(5,2,3),(5,3,2),(5,4,1)}
