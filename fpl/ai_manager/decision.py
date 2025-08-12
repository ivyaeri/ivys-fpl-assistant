# fpl/ai_manager/decision.py
import re, json, streamlit as st, pandas as pd
from langchain_openai import ChatOpenAI
from fpl.ai_manager.core import (
    build_scored_market,
    pick_starting_xi,
    simulate_week_points,
    ensure_auto_state,  # used as fallback when AI draft fails
)

# ------------ helpers ------------
def _safe_json(s: str) -> dict:
    m = re.search(r'\{.*\}', s, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

def _extract_json_obj(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

# ------------ AI weekly decision (transfer/hold + chip + captain) ------------
def decide_with_ai(state: dict, market_df: pd.DataFrame, kb_text: str, api_key: str, model_name: str, gw_now: int) -> dict:
    llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.2)
    squad_ids = [p["id"] for p in state["squad"]]
    squad_txt = market_df[market_df["id"].isin(squad_ids)][
        ["id","web_name","team_short","pos","price","form","status","selected_by","points_per_game"]
    ].sort_values(["pos","web_name"]).to_string(index=False)
    chip_avail = [k for k,v in state.get("chips",{}).items() if v] or ["NONE"]
    system = "You are an autonomous FPL manager. Return STRICT JSON only with required keys."
    user = f"""
Gameweek: {gw_now}
Free transfers: {state['free_transfers']}
Bank: £{state['bank']:.1f}m
Available chips: {chip_avail} (use at most one)

CURRENT 15:
{squad_txt}

KNOWLEDGE BASE:
{kb_text}

Return JSON only:
{{
 "chip": "NONE" | "TC" | "BB" | "FH" | "WC",
 "made": true | false,
 "out_id": <int or null>,
 "in_id": <int or null>,
 "captain_id": <int or null>,
 "reason": "<short>"
}}
Constraints: single like-for-like swap unless FH/WC; respect budget and ≤3 per club; choose a sensible captain; play TC on elite doubles, BB with strong bench.
"""
    raw = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}]).content.strip()
    dec = _safe_json(raw) or {"chip":"NONE","made":False,"out_id":None,"in_id":None,"captain_id":None,"reason":"Parse failed; hold."}
    if dec.get("chip") not in {"NONE","TC","BB","FH","WC"}: dec["chip"]="NONE"
    if not isinstance(dec.get("made"), bool): dec["made"]=False
    return dec

def apply_transfer_if_legal(state: dict, market: pd.DataFrame, decision: dict) -> tuple[bool, str]:
    if not decision.get("made"): return False, "Hold chosen."
    if state["free_transfers"] < 1: return False, "No free transfers."
    out_id, in_id = decision.get("out_id"), decision.get("in_id")
    if not out_id or not in_id: return False, "Missing ids."
    squad_ids = [s["id"] for s in state["squad"]]
    if out_id not in squad_ids or in_id in squad_ids: return False, "Invalid id(s)."
    out_row = market.loc[market["id"] == int(out_id)]
    in_row  = market.loc[market["id"] == int(in_id)]
    if out_row.empty or in_row.empty: return False, "Player not in market."
    if out_row.iloc[0]["pos"] != in_row.iloc[0]["pos"]: return False, "Not like-for-like."
    delta = float(in_row.iloc[0]["price"]) - float(out_row.iloc[0]["price"])
    if delta > state["bank"] + 1e-6: return False, "Over budget."
    temp_ids = [sid for sid in squad_ids if sid != int(out_id)] + [int(in_id)]
    temp_sq = market[market["id"].isin(temp_ids)]
    if temp_sq.groupby("team_short")["id"].count().max() > 3: return False, "Exceeds 3-per-club."
    # apply
    for i,p in enumerate(state["squad"]):
        if p["id"] == int(out_id):
            state["squad"][i] = {"id": int(in_id), "buy_price": float(in_row.iloc[0]["price"])}
            break
    state["bank"] = float(state["bank"] - max(0.0, delta))
    state["free_transfers"] = max(0, state["free_transfers"] - 1)
    return True, "Applied."

def run_ai_auto_until_current(kb_meta: dict, players_df: pd.DataFrame, teams_df: pd.DataFrame, fixtures: list, fetch_player_history, model_name: str):
    if "auto_mgr" not in st.session_state: return
    state = st.session_state.auto_mgr
    gw_now = kb_meta.get("gw")
    if not gw_now: return
    if state["last_gw_processed"] is None:
        state["last_gw_processed"] = gw_now - 1

    # scored market for XI/captain; AI still decides transfers/chips
    market = build_scored_market(players_df, n_fixt=3)
    api_key = st.session_state.openai_key

    for gw in range(state["last_gw_processed"] + 1, gw_now + 1):
        state["free_transfers"] = min(2, state["free_transfers"] + 1)

        if api_key:
            ai_dec = decide_with_ai(state, market, kb_meta.get("header","") + "\n\n" + st.session_state.full_kb, api_key, model_name, gw)
        else:
            ai_dec = {"chip":"NONE","made":False,"out_id":None,"in_id":None,"captain_id":None,"reason":"No API key; hold."}

        chip = ai_dec.get("chip","NONE")
        if chip in ("TC","BB") and not state["chips"].get(chip, False):
            chip = "NONE"

        made, msg = (False, "")
        reason = ai_dec.get("reason","")

        if ai_dec.get("made") and chip not in ("FH","WC"):
            made, msg = apply_transfer_if_legal(state, market, ai_dec)
            if not made:
                reason = (reason + f" ({msg})").strip()
        elif ai_dec.get("made") and chip in ("FH","WC"):
            reason = (reason + f" (Chip '{chip}' not executed in this build.)").strip()
            chip = "NONE"

        squad_ids = [s["id"] for s in state["squad"]]
        xi_df, meta = pick_starting_xi(market[market["id"].isin(squad_ids)].copy())
        captain_id = int(ai_dec.get("captain_id") or meta["captain_id"])
        if chip == "TC" and captain_id not in xi_df["id"].tolist():
            captain_id = int(meta["captain_id"])
        bench_ids = meta.get("bench_ids", [])

        pts = simulate_week_points(xi_df, captain_id, gw, chip=chip, bench_ids=bench_ids)

        used_chip = chip if chip in ("TC","BB") else "NONE"
        if used_chip in ("TC","BB"):
            state["chips"][used_chip] = False

        state["log"].append({
            "gw": int(gw),
            "transfer": {"out": ai_dec.get("out_id"), "in": ai_dec.get("in_id")} if made else None,
            "made": bool(made),
            "reason": reason or msg or "—",
            "chip": used_chip,
            "xi_ids": xi_df["id"].tolist(),
            "bench_ids": bench_ids,
            "captain_id": int(captain_id),
            "points": int(pts),
            "bank": float(state["bank"]),
            "free_transfers": int(state["free_transfers"]),
            "squad_ids": squad_ids,
        })

        state["last_gw_processed"] = gw

# ------------ NEW: AI initial (GW1) squad draft ------------
def _validate_initial(ids: list[int], players_df: pd.DataFrame, budget: float = 100.0) -> tuple[bool, str]:
    if not isinstance(ids, list) or len(ids) != 15:
        return False, "Need exactly 15 ids."
    ids = [int(x) for x in ids]
    if len(set(ids)) != 15:
        return False, "Duplicate ids."
    sub = players_df[players_df["id"].isin(ids)].copy()
    if len(sub) != 15:
        return False, "Unknown ids."
    need = {"GK":2,"DEF":5,"MID":5,"FWD":3}
    got = sub["pos"].value_counts().to_dict()
    for k,v in need.items():
        if got.get(k,0) != v:
            return False, f"Wrong shape: {got}."
    cost = float(sub["price"].sum())
    if cost > budget + 1e-6:
        return False, f"Over budget: £{cost:.1f}m > £{budget:.1f}m."
    if sub["team_short"].value_counts().max() > 3:
        return False, "Exceeds 3-per-club."
    return True, ""

def ensure_initial_squad_with_ai(players_df: pd.DataFrame, kb_text: str, api_key: str, model_name: str, budget: float = 100.0):
    """If there's no squad, ask the LLM to draft a full 15. On failure/no key, fallback to greedy."""
    if "auto_mgr" in st.session_state and st.session_state.auto_mgr.get("squad"):
        return

    # No API? fall back to greedy
    if not api_key:
        ensure_auto_state(players_df, None)
        st.session_state.auto_mgr["seed_origin"] = "greedy_no_api"
        return

    llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.1)
    market_txt = players_df[["id","web_name","team_short","pos","price","form","status","selected_by"]].to_string(index=False)
    system = "You are drafting a starting 15-man FPL squad. Return STRICT JSON only."
    user = f"""
Budget: £{budget:.1f}m. Shape must be exactly: GK=2, DEF=5, MID=5, FWD=3. Max 3 from any club.
Use the knowledge base and this player table to choose IDs.

PLAYERS:
{market_txt}

KNOWLEDGE BASE:
{kb_text}

Return JSON ONLY:
{{
  "squad_ids": [15 unique player ids],
  "reason": "<short draft rationale>",
  "captain_id": <int or null>
}}
"""
    raw = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}]).content.strip()
    obj = _extract_json_obj(raw) or {}
    squad_ids = obj.get("squad_ids") or []

    ok, why = _validate_initial(squad_ids, players_df, budget=budget)
    if not ok:
        ensure_auto_state(players_df, None)
        st.session_state.auto_mgr["seed_origin"] = f"greedy_fallback ({why or 'parse_failed'})"
        return

    sub = players_df[players_df["id"].isin([int(x) for x in squad_ids])][["id","price"]].copy()
    st.session_state.auto_mgr = {
        "squad": [{"id": int(r.id), "buy_price": float(r.price)} for r in sub.itertuples(index=False)],
        "bank": float(budget - sub["price"].sum()),
        "free_transfers": 1,
        "last_gw_processed": None,   # so the runner will process GW1 next
        "chips": {"TC": True, "BB": True, "FH": True, "WC1": True, "WC2": True},
        "log": [],
        "seed_origin": "ai",
        "seed_reason": obj.get("reason",""),
    }

# ------------ NEW: Rewind & regenerate current GW ------------
def rewind_and_regenerate_current_gw(kb_meta: dict, players_df: pd.DataFrame, teams_df: pd.DataFrame, fixtures: list, fetch_player_history, model_name: str):
    """Remove current GW entry (if any) and rerun AI for it."""
    if "auto_mgr" not in st.session_state:
        return False, "No state."
    state = st.session_state.auto_mgr
    gw_now = kb_meta.get("gw")
    if not gw_now:
        return False, "No current GW."

    before = len(state["log"])
    state["log"] = [e for e in state["log"] if int(e.get("gw", -1)) != int(gw_now)]
    state["last_gw_processed"] = gw_now - 1

    run_ai_auto_until_current(
        kb_meta=kb_meta,
        players_df=players_df,
        teams_df=teams_df,
        fixtures=fixtures,
        fetch_player_history=fetch_player_history,
        model_name=model_name,
    )
    after = len(state["log"])
    return after > before, "Regenerated." if after > before else "No change."
