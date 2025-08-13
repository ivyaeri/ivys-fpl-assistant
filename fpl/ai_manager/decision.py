# fpl/ai_manager/decision.py
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from fpl.api import fetch_player_history
from fpl.ai_manager.core import SQUAD_SHAPE, MAX_PER_CLUB, VALID_FORMATIONS
from fpl.ai_manager.persist_db import save_state, append_gw_log

import re, json
# ---------- utils ----------
def _json_from_text(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _validate_initial(players_df: pd.DataFrame, ids: list[int], budget: float = 100.0) -> tuple[bool,str]:
    if not isinstance(ids, list) or len(ids) != 15:
        return False, "Need 15 ids."
    ids = [int(x) for x in ids]
    if len(set(ids)) != 15:
        return False, "Duplicate ids."
    sub = players_df[players_df["id"].isin(ids)].copy()
    if len(sub) != 15:
        return False, "Unknown ids."

    shape = sub["pos"].value_counts().to_dict()
    for p, need in SQUAD_SHAPE.items():
        if shape.get(p, 0) != need:
            return False, f"Wrong shape: {shape}."

    if float(sub["price"].sum()) > budget + 1e-6:
        return False, "Over budget."
    if sub["team_short"].value_counts().max() > MAX_PER_CLUB:
        return False, "Exceeds 3/club."
    return True, ""

def _validate_lineup(players_df: pd.DataFrame, squad_ids: list[int], xi_ids: list[int], bench_order: list[int]) -> tuple[bool,str]:
    all_ids = set(squad_ids)
    xi = list(map(int, xi_ids or []))
    bench = list(map(int, bench_order or []))
    if len(xi) != 11:
        return False, "XI must have 11 ids."
    if len(bench) != 4:
        return False, "Bench must have 4 ids."
    if set(xi) & set(bench):
        return False, "XI and bench overlap."
    if set(xi) | set(bench) != all_ids:
        return False, "XI+bench must cover all 15."

    sub = players_df[players_df["id"].isin(xi)]
    counts = sub["pos"].value_counts().to_dict()
    defc, midc, fwdc = counts.get("DEF", 0), counts.get("MID", 0), counts.get("FWD", 0)
    if (defc, midc, fwdc) not in VALID_FORMATIONS:
        return False, f"Invalid formation DEF-MID-FWD: {(defc, midc, fwdc)}."
    if counts.get("GK", 0) != 1:
        return False, "XI must have exactly 1 GK."
    return True, ""

def _validate_transfer(players_df: pd.DataFrame, squad_ids: list[int], bank: float,
                       out_id: int | None, in_id: int | None) -> tuple[bool,str,float,list[int]]:
    if out_id is None and in_id is None:
        return True, "Hold.", bank, squad_ids
    if out_id is None or in_id is None:
        return False, "Missing ids.", bank, squad_ids

    out_id, in_id = int(out_id), int(in_id)
    if out_id not in squad_ids:
        return False, "Out id not in squad.", bank, squad_ids
    if in_id in squad_ids:
        return False, "In id already in squad.", bank, squad_ids

    out = players_df.loc[players_df["id"] == out_id]
    inn = players_df.loc[players_df["id"] == in_id]
    if out.empty or inn.empty:
        return False, "Unknown id(s).", bank, squad_ids
    if out.iloc[0]["pos"] != inn.iloc[0]["pos"]:
        return False, "Must be like-for-like.", bank, squad_ids

    tmp = players_df[players_df["id"].isin([sid for sid in squad_ids if sid != out_id] + [in_id])]
    if tmp["team_short"].value_counts().max() > MAX_PER_CLUB:
        return False, "Would exceed 3/club.", bank, squad_ids

    delta = float(inn.iloc[0]["price"]) - float(out.iloc[0]["price"])
    if delta > bank + 1e-6:
        return False, "Over budget.", bank, squad_ids

    new_bank = bank - float(delta)
    new_squad = [sid for sid in squad_ids if sid != out_id] + [in_id]
    return True, "Applied.", new_bank, new_squad

def _event_points(pid: int, gw: int) -> int:
    try:
        h = fetch_player_history(pid).get("history", [])
        for g in h:
            if int(g.get("round", -1)) == int(gw):
                return int(g.get("total_points", 0))
    except Exception:
        pass
    return 0

def _compute_points(xi_ids: list[int], cap_id: int, bench_ids: list[int], gw: int, chip: str) -> int:
    xi_pts = sum(_event_points(int(pid), gw) for pid in xi_ids)
    cap_pts = _event_points(int(cap_id), gw) if cap_id else 0
    total = xi_pts + cap_pts
    if chip == "TC":
        total += cap_pts  # triple captain adds +1x captain points (since we already counted him twice)
    if chip == "BB":
        total += sum(_event_points(int(pid), gw) for pid in bench_ids)
    return int(total)

def _llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=st.session_state.openai_key, model_name=model_name, temperature=0.2)

# ---------- prompts ----------
def draft_initial_squad(players_df: pd.DataFrame, kb_text: str, model_name: str, budget: float = 100.0) -> dict:
    if not st.session_state.openai_key:
        return {"error": "no_api"}
    llm = _llm(model_name)
    table = players_df[["id","web_name","team_short","pos","price","form","status","selected_by"]].to_string(index=False)
    # --- replace your current `sys` and `usr` in draft_initial_squad(...) with this ---

    sys = """
    You are an elite Fantasy Premier League drafter. You will select a legal 15-man FPL squad for GW1
    using only the tables and knowledge base provided. You MUST obey all constraints and return
    STRICT JSON ONLY — no prose, no markdown, no code fences.
    
    Hard rules:
    - Total budget must be within the user’s budget.
    - Exactly 15 players with shape: GK=2, DEF=5, MID=5, FWD=3.
    - Max 3 players per club.
    - Only pick players that appear in the PLAYERS table.
    - Prefer status 'a' (Available); avoid injured/suspended. If you choose a flagged player, justify it.
    - Output must be exactly: {"squad_ids":[...], "captain_id": <int|null>, "reason":"..."} with integers only.
    - No extra keys, no comments, no trailing commas.
    """

    usr = f"""
    Context:
    - Budget: £{budget:.1f}m
    - Required shape: GK=2, DEF=5, MID=5, FWD=3 (exact)
    - Club cap: ≤ 3 per club
    - Selection signals to consider (from the data below): form, points_per_game, minutes (reliability),
      status/news (injury/rotation risk), ownership (template vs differential), and next fixtures (difficulty).
    
    PLAYERS (id, name, team, pos, price, form, status, selected_by, etc.):
    {table}
    
    KNOWLEDGE BASE (fixtures + player lines):
    {kb_text}
    
    Drafting guidance (apply judgement, but follow the rules above):
    - Balance safe “template” picks with a few value differentials (generally <10% owned) if fixtures/form justify it.
    - Give weight to good near-term fixtures; avoid clusters of tough fixtures at the same position when possible.
    - Prefer nailed starters (high recent minutes) over rotation risks.
    - Avoid players with negative injury/suspension news; if you include one, explain why in the reason.
    - Choose a captain with high xGoal involvement proxies: strong fixture, likely 90 minutes, penalties/set pieces if indicated,
      recent form and high points_per_game.
    - Bench strategy: include budget enablers who are likely to play; ensure at least two playable DEF and one playable MID/FWD on bench.
    
    Return JSON ONLY (no extra keys, integers for IDs):
    {{
      "squad_ids": [15 integer ids],
      "captain_id": <integer id or null>,
      "reason": "<120–220 words explaining the squad structure, key picks, captain choice, and any notable risks>"
    }}
    """

    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    return _json_from_text(raw) or {"error":"parse"}

def weekly_decision(
    players_df: pd.DataFrame,
    kb_text: str,
    state: dict,
    model_name: str,
    gw: int,
    extra_instructions: str | None = None,   # optional manager note for this run
) -> dict:
    if not st.session_state.openai_key:
        return {"error":"no_api"}
    llm = _llm(model_name)
    squad_ids = state["squad"]
    sub = players_df[players_df["id"].isin(squad_ids)][
        ["id","web_name","team_short","pos","price","status","form","points_per_game"]
    ].sort_values(["pos","web_name"])
    table = sub.to_string(index=False)
    chips = [k for k,v in state.get("chips",{}).items() if v] or ["NONE"]

    note = (extra_instructions or "").strip()
    if note:
        note = note[:800]

    sys = "You are an autonomous FPL manager. Return STRICT JSON only."
    usr = f"""
GW {gw}. Free transfers: {state['free_transfers']}. Bank £{state['bank']:.1f}m. Chips available: {chips}.

CURRENT 15:
{table}

KB:
{kb_text}
"""
    if note:
        usr += f"\nMANAGER INSTRUCTIONS (user-provided):\n{note}\n"

    usr += """
Choose AT MOST one transfer, and optionally one chip (TC or BB; FH/WC unsupported here).
Pick a valid XI, bench order (4 ids), and a captain in the XI.

Return JSON ONLY:
{
  "made": true|false,
  "out_id": <int|null>,
  "in_id": <int|null>,
  "chip": "NONE"|"TC"|"BB",
  "xi_ids": [11 ids],
  "bench_order": [4 ids],
  "captain_id": <int>,
  "reason": "<short>"
}
Rules: like-for-like swap; stay under budget and ≤3 per club; XI must have 1 GK and a legal FPL formation; bench has remaining 4 players.
"""
    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    return _json_from_text(raw) or {"error":"parse"}

# ---------- orchestration ----------
def ensure_initial_squad_with_ai(user_id: str, players_df: pd.DataFrame, kb_text: str,
                                 model_name: str, budget: float = 100.0):
    """If no squad, ask LLM to draft one. No greedy fallback."""
    if "auto_mgr" in st.session_state and st.session_state.auto_mgr.get("squad"):
        return
    obj = draft_initial_squad(players_df, kb_text, model_name, budget=budget)

    # error / no-api path
    if obj.get("error"):
        st.session_state.auto_mgr = {
            "squad": [],
            "bank": budget,
            "free_transfers": 0,
            "last_gw_processed": None,
            "last_ft_accrual_gw": 0,  # NEW: accrual guard
            "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
            "log": [],
            "seed_origin": obj["error"],
        }
        save_state(user_id, st.session_state.auto_mgr)
        return

    # validate draft
    ids = obj.get("squad_ids") or []
    ok, why = _validate_initial(players_df, ids, budget)
    if not ok:
        st.session_state.auto_mgr = {
            "squad": [],
            "bank": budget,
            "free_transfers": 0,
            "last_gw_processed": None,
            "last_ft_accrual_gw": 0,  # NEW
            "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
            "log": [],
            "seed_origin": f"ai_failed:{why}",
        }
        save_state(user_id, st.session_state.auto_mgr)
        return

    cost = float(players_df[players_df["id"].isin(ids)]["price"].sum())
    st.session_state.auto_mgr = {
        "squad": list(map(int, ids)),
        "bank": float(budget - cost),
        "free_transfers": 0,              # FPL starts with 1 FT
        "last_gw_processed": None,
        "last_ft_accrual_gw": 0,          # NEW: no accrual performed yet
        "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
        "log": [],
        "seed_origin": "ai",
        "seed_reason": obj.get("reason",""),
    }
    save_state(user_id, st.session_state.auto_mgr)

def run_ai_auto_until_current(user_id: str, kb_meta: dict, players_df: pd.DataFrame,
                              model_name: str, extra_instructions: str | None = None):
    """
    Advance from last_gw_processed+1 → current GW.
    FT accrual happens at the START of each GW (except GW1) and only once per GW.
    """
    if "auto_mgr" not in st.session_state:
        return
    state = st.session_state.auto_mgr
    gw_now = kb_meta.get("gw")
    if not gw_now or not state.get("squad"):
        return

    if state.get("last_gw_processed") is None:
        state["last_gw_processed"] = int(gw_now) - 1

    # backward compatibility for older saves
    state.setdefault("last_ft_accrual_gw", 0)

    for gw in range(int(state["last_gw_processed"]) + 1, int(gw_now) + 1):
        if not st.session_state.openai_key:
            break

        # ✅ ACCRUE FT AT START (not GW1) and only once per GW
        if gw > 1 and state.get("last_ft_accrual_gw") != gw:
            state["free_transfers"] = min(5, state["free_transfers"] + 1)
            state["last_ft_accrual_gw"] = gw

        dec = weekly_decision(
            players_df,
            st.session_state.full_kb,
            state,
            model_name,
            gw,
            extra_instructions=extra_instructions if gw == gw_now else None,  # only apply to this run's current GW
        )
        if dec.get("error"):
            break

        made = bool(dec.get("made", False))
        out_id, in_id = dec.get("out_id"), dec.get("in_id")
        ok, msg, new_bank, new_squad = _validate_transfer(players_df, state["squad"], state["bank"], out_id, in_id)
        if made and not ok:
            # reject this week; don't log incomplete decision
            break
        if made and ok:
            state["squad"] = new_squad
            state["bank"] = float(new_bank)
            state["free_transfers"] = max(0, state["free_transfers"] - 1)

        xi_ids = list(map(int, dec.get("xi_ids") or []))
        bench_order = list(map(int, dec.get("bench_order") or dec.get("bench_ids") or []))
        ok, why = _validate_lineup(players_df, state["squad"], xi_ids, bench_order)
        if not ok:
            break

        cap_id = int(dec.get("captain_id") or 0)
        if cap_id not in xi_ids:
            break

        chip = dec.get("chip", "NONE")
        if chip not in ("NONE", "TC", "BB"):
            chip = "NONE"
        if chip in ("TC", "BB") and not state["chips"].get(chip, False):
            chip = "NONE"

        pts = _compute_points(xi_ids, cap_id, bench_order, gw, chip)

        used_chip = chip
        if used_chip in ("TC", "BB"):
            state["chips"][used_chip] = False

        entry = {
            "gw": int(gw),
            "made": bool(made),
            "transfer": {"out": int(out_id) if out_id else None, "in": int(in_id) if in_id else None} if made else None,
            "chip": used_chip,
            "xi_ids": xi_ids,
            "bench_ids": bench_order,
            "captain_id": cap_id,
            "points": int(pts),
            "bank": float(state["bank"]),
            "free_transfers": int(state["free_transfers"]),  # value AFTER this GW’s decision
            "squad_ids": list(map(int, state["squad"])),
            "reason": dec.get("reason", ""),
        }
        state["log"].append(entry)
        state["last_gw_processed"] = gw

        save_state(user_id, state)
        append_gw_log(user_id, gw, entry)

def rewind_and_regenerate_current_gw(user_id: str, kb_meta: dict, players_df: pd.DataFrame,
                                     model_name: str, extra_instructions: str | None = None):
    """Set pointer back one and re-run a single GW (current), with optional user note."""
    if "auto_mgr" not in st.session_state:
        return False, "No state."
    state = st.session_state.auto_mgr
    gw_now = kb_meta.get("gw")
    if not gw_now:
        return False, "No current GW."
    if not state.get("squad"):
        return False, "No squad."

    # Remove in-memory log for gw_now (DB history is immutable)
    state["log"] = [e for e in state["log"] if int(e.get("gw", -1)) != int(gw_now)]
    state["last_gw_processed"] = int(gw_now) - 1
    # DO NOT touch 'last_ft_accrual_gw' — guard prevents double accrual
    save_state(user_id, state)

    run_ai_auto_until_current(
        user_id=user_id,
        kb_meta=kb_meta,
        players_df=players_df,
        model_name=model_name,
        extra_instructions=extra_instructions,
    )
    return True, "Regenerated."
def refresh_logged_points(user_id: str) -> int:
    """Recompute points for all logged GWs from official FPL history."""
    if "auto_mgr" not in st.session_state:
        return 0
    state = st.session_state.auto_mgr
    updated = 0
    for entry in state.get("log", []):
        gw = int(entry["gw"])
        xi_ids = list(map(int, entry.get("xi_ids", [])))
        bench_ids = list(map(int, entry.get("bench_ids") or entry.get("bench_order") or []))
        cap_id = int(entry.get("captain_id") or 0)
        chip = entry.get("chip", "NONE")
        new_pts = _compute_points(xi_ids, cap_id, bench_ids, gw, chip)
        if new_pts != entry.get("points"):
            entry["points"] = int(new_pts)
            append_gw_log(user_id, gw, entry)  # upsert same PK (user_id, season, gw)
            updated += 1
    save_state(user_id, state)
    return updated
