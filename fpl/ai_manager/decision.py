# fpl/ai_manager/decision.py
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from fpl.api import fetch_player_history
from fpl.ai_manager.core import SQUAD_SHAPE, MAX_PER_CLUB, VALID_FORMATIONS
from fpl.ai_manager.persist_db import save_state, append_gw_log

import re, json
from typing import Tuple, List, Dict, Any
from copy import deepcopy
from datetime import datetime, timezone
import typing as _t

# ---------- utils ----------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _gw_deadline_utc(kb_meta: dict, gw: int) -> _t.Optional[datetime]:
    """
    Try to fetch the GW deadline (UTC) from kb_meta. If not present, return None.
    You can optionally add 'deadline_utc' (for current GW) and/or 'deadlines' {gw: iso} in kb_meta.
    """
    if not isinstance(kb_meta, dict):
        return None
    # exact value for current gw
    if kb_meta.get("deadline_utc"):
        try:
            # accept datetime or iso string
            d = kb_meta["deadline_utc"]
            return d if isinstance(d, datetime) else datetime.fromisoformat(str(d))
        except Exception:
            pass
    # map of all deadlines
    dmap = kb_meta.get("deadlines") or kb_meta.get("deadline_by_gw") or {}
    if gw in dmap:
        try:
            d = dmap[gw]
            return d if isinstance(d, datetime) else datetime.fromisoformat(str(d))
        except Exception:
            return None
    return None

def _ensure_checkpoint(state: dict, gw: int, kb_meta: dict) -> None:
    """
    Save a pre-decision snapshot for this GW (after FT accrual, before any transfers).
    """
    state.setdefault("checkpoints", {})
    key = str(int(gw))
    if key in state["checkpoints"]:
        return  # already saved for this GW
    snap = {
        "gw": int(gw),
        "when_utc": _utc_now().isoformat(),
        "deadline_utc": (_gw_deadline_utc(kb_meta, gw) or None).isoformat() if _gw_deadline_utc(kb_meta, gw) else None,
        "squad": list(map(int, state.get("squad", []))),
        "bank": float(state.get("bank", 0.0)),
        "free_transfers": int(state.get("free_transfers", 0)),
        "chips": deepcopy(state.get("chips", {})),
        "last_ft_accrual_gw": int(state.get("last_ft_accrual_gw", 0)),
    }
    state["checkpoints"][key] = snap

def _restore_checkpoint_if_allowed(state: dict, gw: int, kb_meta: dict) -> tuple[bool, str]:
    """
    Restore the saved pre-decision snapshot for this GW if we're still before the deadline.
    If kb_meta has no deadline, we allow restore (best effort).
    """
    snap = (state.get("checkpoints") or {}).get(str(int(gw)))
    if not snap:
        return False, "No checkpoint for this GW."

    # enforce deadline if we have it
    d_iso = snap.get("deadline_utc")
    if d_iso:
        try:
            d = datetime.fromisoformat(d_iso)
            if _utc_now() > d:
                return False, "Deadline passed; keeping committed state."
        except Exception:
            pass  # if parsing fails, fall through and allow restore

    # restore fields
    state["squad"] = list(map(int, snap.get("squad", [])))
    state["bank"] = float(snap.get("bank", 0.0))
    state["free_transfers"] = int(snap.get("free_transfers", 0))
    state["chips"] = deepcopy(snap.get("chips", {}))
    state["last_ft_accrual_gw"] = int(snap.get("last_ft_accrual_gw", 0))
    return True, "Restored pre-decision snapshot."


def _json_from_text(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
def _snapshot(players_df: pd.DataFrame, codes: list[int]) -> list[dict]:
    """Return a compact, UI-friendly snapshot for the given 15/11/bench (by CODE)."""
    if not codes:
        return []
    cols = ["code","web_name","team_short","pos","price","status","form","points_per_game"]
    missing = [c for c in ["code","web_name","team_short","pos","price"] if c not in players_df.columns]
    if missing:
        return []
    sub = players_df[players_df["code"].isin(list(map(int, codes)))].copy()
    if sub.empty:
        return []
    keep = [c for c in cols if c in sub.columns]
    sub = sub[keep].sort_values(["pos","web_name"])
    # records for Streamlit UI
    return sub.to_dict(orient="records")
def _ensure_maps(players_df: pd.DataFrame) -> Tuple[Dict[int,int], Dict[int,int]]:
    """
    Returns (code_to_id, id_to_code) from players_df (expects both 'code' and 'id' cols).
    """
    if "code" not in players_df.columns:
        raise ValueError("players_df must contain a 'code' column.")
    if "id" in players_df.columns:
        code_to_id = {int(c): int(i) for c, i in zip(players_df["code"], players_df["id"])}
        id_to_code = {int(i): int(c) for i, c in zip(players_df["id"], players_df["code"])}
    else:
        code_to_id, id_to_code = {}, {}
    return code_to_id, id_to_code

# --- Compat: migrate old ID-based state/logs -> CODE-based --------------------
def _build_maps(players_df: pd.DataFrame):
    id_to_code = {}
    code_to_id = {}
    if "id" in players_df.columns and "code" in players_df.columns:
        id_to_code = {int(i): int(c) for i, c in zip(players_df["id"], players_df["code"])}
        code_to_id = {int(c): int(i) for i, c in zip(players_df["code"], players_df["id"])}
    return id_to_code, code_to_id

def _ids_list_to_codes(lst, id_to_code):
    out = []
    for x in lst or []:
        try:
            xi = int(x)
        except Exception:
            continue
        out.append(int(id_to_code.get(xi, xi)))
    return out

def _maybe_migrate_state_to_codes(state: dict, players_df: pd.DataFrame) -> bool:
    """Return True if we changed the state."""
    if not state or "squad" not in state:
        return False
    id_to_code, _ = _build_maps(players_df)
    if not id_to_code:
        return False

    squad = list(map(int, state.get("squad") or []))
    have_any_id = "id" in players_df.columns and any(int(x) in set(players_df["id"]) for x in squad)
    have_any_code = any(int(x) in set(players_df["code"]) for x in squad)
    if have_any_id and not have_any_code:
        state["squad"] = _ids_list_to_codes(squad, id_to_code)

        new_logs = []
        for e in state.get("log", []):
            e = dict(e)
            if "xi_ids" in e and "xi_codes" not in e:
                e["xi_codes"] = _ids_list_to_codes(e["xi_ids"], id_to_code)
            if "bench_ids" in e and "bench_codes" not in e:
                e["bench_codes"] = _ids_list_to_codes(e["bench_ids"], id_to_code)
            if "bench_order" in e and "bench_codes" not in e:
                e["bench_codes"] = _ids_list_to_codes(e["bench_order"], id_to_code)
            if "captain_id" in e and "captain_code" not in e:
                cap = int(e["captain_id"]) if e["captain_id"] is not None else 0
                e["captain_code"] = int(id_to_code.get(cap, cap)) if cap else 0
            if "transfers" in e:
                fixed = []
                for t in e["transfers"]:
                    t = dict(t)
                    if "out_code" not in t and "out_id" in t:
                        t["out_code"] = int(id_to_code.get(int(t["out_id"]), t["out_id"]))
                    if "in_code" not in t and "in_id" in t:
                        t["in_code"] = int(id_to_code.get(int(t["in_id"]), t["in_id"]))
                    fixed.append(t)
                e["transfers"] = fixed
            if "squad_ids" in e and "squad_codes" not in e:
                e["squad_codes"] = _ids_list_to_codes(e["squad_ids"], id_to_code)
            new_logs.append(e)
        state["log"] = new_logs
        return True
    return False

# ---------- validation (CODES, not ids) ----------
def _validate_initial(players_df: pd.DataFrame, codes: List[int], budget: float = 100.0) -> Tuple[bool,str]:
    if not isinstance(codes, list) or len(codes) != 15:
        return False, "Need 15 codes."
    codes = [int(x) for x in codes]
    if len(set(codes)) != 15:
        return False, "Duplicate codes."
    sub = players_df[players_df["code"].isin(codes)].copy()
    if len(sub) != 15:
        return False, "Unknown codes."

    shape = sub["pos"].value_counts().to_dict()
    for p, need in SQUAD_SHAPE.items():
        if shape.get(p, 0) != need:
            return False, f"Wrong shape: {shape}."

    if float(sub["price"].sum()) > budget + 1e-6:
        return False, "Over budget."
    if sub["team_short"].value_counts().max() > MAX_PER_CLUB:
        return False, "Exceeds 3/club."
    return True, ""

def _validate_lineup(
    players_df: pd.DataFrame,
    squad_codes: List[int],
    xi_codes: List[int],
    bench_codes: List[int],
) -> Tuple[bool,str]:
    all_codes = set(map(int, squad_codes))
    xi = list(map(int, xi_codes or []))
    bench = list(map(int, bench_codes or []))
    if len(xi) != 11:
        return False, "XI must have 11 codes."
    if len(bench) != 4:
        return False, "Bench must have 4 codes."
    if set(xi) & set(bench):
        return False, "XI and bench overlap."
    if set(xi) | set(bench) != all_codes:
        return False, "XI+bench must cover all 15."

    sub = players_df[players_df["code"].isin(xi)]
    counts = sub["pos"].value_counts().to_dict()
    defc, midc, fwdc = counts.get("DEF", 0), counts.get("MID", 0), counts.get("FWD", 0)
    if (defc, midc, fwdc) not in VALID_FORMATIONS:
        return False, f"Invalid formation DEF-MID-FWD: {(defc, midc, fwdc)}."
    if counts.get("GK", 0) != 1:
        return False, "XI must have exactly 1 GK."
    return True, ""

def _validate_transfer(
    players_df: pd.DataFrame,
    squad_codes: List[int],
    bank: float,
    out_code: int | None,
    in_code: int | None,
) -> Tuple[bool,str,float,List[int]]:
    """Single transfer validator (codes)."""
    if out_code is None and in_code is None:
        return True, "Hold.", bank, squad_codes
    if out_code is None or in_code is None:
        return False, "Missing codes.", bank, squad_codes

    out_code, in_code = int(out_code), int(in_code)
    if out_code not in squad_codes:
        return False, "Out code not in squad.", bank, squad_codes
    if in_code in squad_codes:
        return False, "In code already in squad.", bank, squad_codes

    out = players_df.loc[players_df["code"] == out_code]
    inn = players_df.loc[players_df["code"] == in_code]
    if out.empty or inn.empty:
        return False, "Unknown code(s).", bank, squad_codes
    if out.iloc[0]["pos"] != inn.iloc[0]["pos"]:
        return False, "Must be like-for-like.", bank, squad_codes

    tmp = players_df[players_df["code"].isin([c for c in squad_codes if c != out_code] + [in_code])]
    if tmp["team_short"].value_counts().max() > MAX_PER_CLUB:
        return False, "Would exceed 3/club.", bank, squad_codes

    delta = float(inn.iloc[0]["price"]) - float(out.iloc[0]["price"])
    if delta > bank + 1e-6:
        return False, "Over budget.", bank, squad_codes

    new_bank = bank - float(delta)
    new_squad = [c for c in squad_codes if c != out_code] + [in_code]
    return True, "Applied.", new_bank, new_squad

# ---------- multi-transfer validator (codes) ----------
HIT_COST_DEFAULT = 4  # points per extra transfer

def _validate_transfers(
    players_df: pd.DataFrame,
    squad_codes: List[int],
    bank: float,
    transfers: List[dict],
) -> Tuple[bool, str, float, List[int]]:
    """
    Apply zero-or-more like-for-like transfers in order (codes).
    transfers: [{"out_code": int, "in_code": int}, ...]
    Returns (ok, msg, new_bank, new_squad_codes).
    """
    if not transfers:
        return True, "Hold.", float(bank), list(squad_codes)

    new_bank = float(bank)
    new_squad = list(map(int, squad_codes))
    seen_out, seen_in = set(), set()

    for t in transfers:
        out_code = t.get("out_code")
        in_code  = t.get("in_code")
        if out_code is None or in_code is None:
            return False, "Transfers must use out_code/in_code.", bank, squad_codes

        try:
            out_code = int(out_code)
            in_code  = int(in_code)
        except Exception:
            return False, "Bad transfer codes.", bank, squad_codes

        if out_code in seen_out:
            return False, f"Duplicate out_code {out_code}.", bank, squad_codes
        if in_code in seen_in:
            return False, f"Duplicate in_code {in_code}.", bank, squad_codes

        if out_code not in new_squad:
            return False, f"Out code {out_code} not in current squad.", bank, squad_codes
        if in_code in new_squad:
            return False, f"In code {in_code} already in squad.", bank, squad_codes

        out = players_df.loc[players_df["code"] == out_code]
        inn = players_df.loc[players_df["code"] == in_code]
        if out.empty or inn.empty:
            return False, "Unknown code(s).", bank, squad_codes

        if out.iloc[0]["pos"] != inn.iloc[0]["pos"]:
            return False, "Must be like-for-like.", bank, squad_codes

        candidate = [c for c in new_squad if c != out_code] + [in_code]
        tmp = players_df[players_df["code"].isin(candidate)]
        if tmp["team_short"].value_counts().max() > MAX_PER_CLUB:
            return False, "Would exceed 3/club.", bank, squad_codes

        delta = float(inn.iloc[0]["price"]) - float(out.iloc[0]["price"])
        if delta > new_bank + 1e-6:
            return False, "Over budget.", bank, squad_codes

        new_bank -= float(delta)
        new_squad = candidate
        seen_out.add(out_code); seen_in.add(in_code)

    return True, "Applied.", float(new_bank), list(new_squad)

# ---------- scoring (codes → id for API) ----------
def _event_points(code: int, gw: int, code_to_id: Dict[int,int]) -> int:
    try:
        pid = code_to_id.get(int(code))
        if not pid:
            return 0
        h = fetch_player_history(int(pid)).get("history", [])
        for g in h:
            if int(g.get("round", -1)) == int(gw):
                return int(g.get("total_points", 0))
    except Exception:
        pass
    return 0

def _compute_points(
    xi_codes: List[int],
    cap_code: int,
    bench_codes: List[int],
    gw: int,
    chip: str,
    code_to_id: Dict[int,int],
) -> int:
    xi_pts = sum(_event_points(int(c), gw, code_to_id) for c in xi_codes)
    cap_pts = _event_points(int(cap_code), gw, code_to_id) if cap_code else 0
    total = xi_pts + cap_pts
    if chip == "TC":
        total += cap_pts  # triple captain adds +1x captain points (since already counted twice)
    if chip == "BB":
        total += sum(_event_points(int(c), gw, code_to_id) for c in bench_codes)
    return int(total)

def _llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=st.session_state.openai_key, model_name=model_name, temperature=0.2)

# ---------- prompts (return CODES) ----------
def draft_initial_squad(
    players_df: pd.DataFrame,
    kb_text: str,
    model_name: str,
    budget: float = 100.0,
    extra_instructions: str | None = None,
    prior_squad_codes: List[int] | None = None,
) -> dict:
    """
    Draft a legal 15-man squad using CODES as identifiers.
    Returns STRICT JSON:
      {"squad_codes":[...], "captain_code": <int|null>, "reason":"..."}
    """
    if not st.session_state.openai_key:
        return {"error": "no_api"}

    if "code" not in players_df.columns:
        return {"error": "players_df missing 'code' column"}

    llm = _llm(model_name)
    table = players_df[
        ["code","web_name","team_short","pos","price","form","status","selected_by","points_per_game"]
    ].sort_values(["pos","web_name"]).to_string(index=False)

    sys = (
        "You are an elite Fantasy Premier League drafter. Always obey constraints and "
        "return STRICT JSON ONLY (no prose/markdown/code fences). Use player CODES."
    )

    prior_block = ""
    if prior_squad_codes:
        prior_sub = players_df[players_df["code"].isin([int(x) for x in prior_squad_codes])][
            ["code","web_name","team_short","pos","price","status","form","points_per_game"]
        ].sort_values(["pos","web_name"]).to_string(index=False)
        prior_block = f"""
PRIOR_SQUAD_CODES: {list(map(int, prior_squad_codes))}
PRIOR_SQUAD_TABLE:
{prior_sub}

If a prior squad is given, start from it and revise MINIMALLY (generally ≤5 swaps) unless
the manager instructions require more. Keep budget/shape/club caps valid. If you believe
no changes are needed, you may return the same 15.
"""

    note_block = f"\nMANAGER INSTRUCTIONS:\n{(extra_instructions or '').strip()[:800]}\n" if extra_instructions else ""

    usr = f"""
Budget: £{budget:.1f}m. Exact shape: GK=2, DEF=5, MID=5, FWD=3. Max 3 per club.
Prefer status 'a' (available). Consider form, points_per_game, minutes reliability,
ownership (template vs differential), and near-term fixtures.

{prior_block}{note_block}
PLAYERS (code, name, team, pos, price, form, status, selected_by, ppg):
{table}

KNOWLEDGE BASE (fixtures + player lines):
{kb_text}

Return JSON ONLY:
{{
  "squad_codes": [15 integer codes],
  "captain_code": <integer code or null>,
  "reason": "<120–220 words on structure, key picks, changes from prior if any>"
}}
Rules:
- Total price ≤ budget; exact 2/5/5/3 shape; ≤3 per club; codes must be from the PLAYERS table.
- If PRIOR_SQUAD_CODES are given, keep changes minimal unless instructions mandate otherwise.
"""

    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    obj = _json_from_text(raw)
    if not obj:
        return {"error":"parse"}

    # Backward-compat: if model accidentally returned ids, map to codes
    if "squad_codes" not in obj and "squad_ids" in obj and "id" in players_df.columns:
        _, id_to_code = _ensure_maps(players_df)
        obj["squad_codes"] = [int(id_to_code.get(int(i), -1)) for i in obj["squad_ids"]]
        obj["captain_code"] = int(id_to_code.get(int(obj.get("captain_id") or 0), 0)) or None

    return obj

def weekly_decision(
    players_df: pd.DataFrame,
    kb_text: str,
    state: dict,
    model_name: str,
    gw: int,
    extra_instructions: str | None = None,
) -> dict:
    """Allow zero-or-more transfers, bench ordering, and chip (TC/BB). Uses CODES."""
    if not st.session_state.openai_key:
        return {"error":"no_api"}
    if "code" not in players_df.columns:
        return {"error":"players_df missing 'code' column"}

    llm = _llm(model_name)
    squad_codes = list(map(int, state["squad"]))
    sub = players_df[players_df["code"].isin(squad_codes)][
        ["code","web_name","team_short","pos","price","status","form","points_per_game"]
    ].sort_values(["pos","web_name"])
    table = sub.to_string(index=False)
    chips = [k for k,v in state.get("chips",{}).items() if v] or ["NONE"]

    note = (extra_instructions or "").strip()
    if note:
        note = note[:800]

    sys = (
        "You are an autonomous FPL manager. "
        "Return STRICT JSON only. No markdown, no comments. Use player CODES."
    )
    usr = f"""
Weekly decision for GW {gw}.

Resources:
- Free transfers available: {state['free_transfers']}
- Bank: £{state['bank']:.1f}m
- Chips available: {chips} (only 'TC' or 'BB' allowed here)
- Constraints: ≤3 per club; stay under budget; like-for-like by position; valid XI (1 GK; allowed formations: {{3-4-3,3-5-2,4-4-2,4-3-3,5-3-2,5-4-1}})
- Pick XI, set bench order (4 codes; 1st = first sub), choose a captain in the XI.

CURRENT 15 (by CODE):
{table}

KNOWLEDGE BASE:
{kb_text}
""" + (f"\nMANAGER NOTE:\n{note}\n" if note else "") + """
Return JSON ONLY (schema EXACTLY):
{
  "chip": "NONE" | "TC" | "BB",
  "transfers": [{"out_code": <int>, "in_code": <int>}],
  "xi_codes": [11 ints],
  "bench_codes": [4 ints],
  "captain_code": <int>,
  "reason": "<short rationale>",
  "transfer_reasons": ["<t1>", "..."],
  "bench_reason": "<why this bench order>"
}
"""
    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    obj = _json_from_text(raw)
    if not obj:
        return {"error":"parse"}

    # Backward-compat guard if model used ids by mistake
    if ("xi_codes" not in obj or "bench_codes" not in obj or "transfers" not in obj) and "id" in players_df.columns:
        _, id_to_code = _ensure_maps(players_df)
        if "xi_codes" not in obj and "xi_ids" in obj:
            obj["xi_codes"] = [int(id_to_code.get(int(i), -1)) for i in obj["xi_ids"]]
        if "bench_codes" not in obj:
            ids = obj.get("bench_ids") or obj.get("bench_order") or []
            obj["bench_codes"] = [int(id_to_code.get(int(i), -1)) for i in ids]
        if "captain_code" not in obj and "captain_id" in obj:
            obj["captain_code"] = int(id_to_code.get(int(obj["captain_id"]), 0)) or None
        tx = obj.get("transfers", [])
        fixed = []
        for t in tx:
            if "out_code" in t and "in_code" in t:
                fixed.append({"out_code": int(t["out_code"]), "in_code": int(t["in_code"])})
            elif "out_id" in t and "in_id" in t:
                fixed.append({
                    "out_code": int(id_to_code.get(int(t["out_id"]), -1)),
                    "in_code": int(id_to_code.get(int(t["in_id"]), -1)),
                })
        obj["transfers"] = fixed

    return obj

# ---------- orchestration (codes) ----------
def ensure_initial_squad_with_ai(
    user_id: str,
    players_df: pd.DataFrame,
    kb_text: str,
    model_name: str,
    budget: float = 100.0,
):
    """If no squad, ask LLM to draft one. Stores CODES in state."""
    # one-time: migrate any legacy id-based state to codes
    if "auto_mgr" in st.session_state and st.session_state.auto_mgr:
        if _maybe_migrate_state_to_codes(st.session_state.auto_mgr, players_df):
            save_state(user_id, st.session_state.auto_mgr)

    if "auto_mgr" in st.session_state and st.session_state.auto_mgr.get("squad"):
        return

    _, id_to_code = _ensure_maps(players_df)

    obj = draft_initial_squad(
        players_df, kb_text, model_name, budget=budget,
        prior_squad_codes=None
    )

    if obj.get("error"):
        st.session_state.auto_mgr = {
            "squad": [],
            "bank": budget,
            "free_transfers": 0,
            "last_gw_processed": None,
            "last_ft_accrual_gw": 0,
            "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
            "log": [],
            "seed_origin": obj["error"],
            "budget": float(budget),
            "hit_cost": HIT_COST_DEFAULT,
            "max_hit": 0,
        }
        save_state(user_id, st.session_state.auto_mgr)
        return

    codes = obj.get("squad_codes") or []
    if (not codes) and obj.get("squad_ids"):
        codes = [int(id_to_code.get(int(i), -1)) for i in obj["squad_ids"]]

    ok, why = _validate_initial(players_df, codes, budget)
    if not ok:
        st.session_state.auto_mgr = {
            "squad": [],
            "bank": budget,
            "free_transfers": 0,
            "last_gw_processed": None,
            "last_ft_accrual_gw": 0,
            "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
            "log": [],
            "seed_origin": f"ai_failed:{why}",
            "budget": float(budget),
            "hit_cost": HIT_COST_DEFAULT,
            "max_hit": 0,
        }
        save_state(user_id, st.session_state.auto_mgr)
        return

    cost = float(players_df[players_df["code"].isin(codes)]["price"].sum())

    st.session_state.auto_mgr = {
        "squad": list(map(int, codes)),
        "bank": float(budget - cost),
        "free_transfers": 0,
        "last_gw_processed": None,
        "last_ft_accrual_gw": 0,
        "chips": {"TC":True,"BB":True,"FH":True,"WC1":True,"WC2":True},
        "log": [],
        "seed_origin": "ai",
        "seed_reason": obj.get("reason",""),
        "budget": float(budget),
        "hit_cost": HIT_COST_DEFAULT,
        "max_hit": 0,
    }
    save_state(user_id, st.session_state.auto_mgr)

def run_ai_auto_until_current(
    user_id: str,
    kb_meta: dict,
    players_df: pd.DataFrame,
    model_name: str,
    extra_instructions: str | None = None
):
    """
    Advance from last_gw_processed+1 → current GW (codes).
    FT accrual happens at the START of each GW (except GW1) and only once per GW.
    """
    if "auto_mgr" not in st.session_state:
        return
    state = st.session_state.auto_mgr

    # one-time compat migration if needed
    if _maybe_migrate_state_to_codes(state, players_df):
        save_state(user_id, state)

    gw_now = kb_meta.get("gw")
    if not gw_now or not state.get("squad"):
        return

    code_to_id = {int(k): int(v) for k, v in (kb_meta.get("code_to_id") or {}).items()}
    if not code_to_id:
        code_to_id, _ = _ensure_maps(players_df)

    if state.get("last_gw_processed") is None:
        state["last_gw_processed"] = int(gw_now) - 1

    state.setdefault("last_ft_accrual_gw", 0)
    state.setdefault("hit_cost", HIT_COST_DEFAULT)
    state.setdefault("max_hit", 0)

    for gw in range(int(state["last_gw_processed"]) + 1, int(gw_now) + 1):
        if not st.session_state.openai_key:
            break

        if gw > 1 and state.get("last_ft_accrual_gw") != gw:
            state["free_transfers"] = min(5, state["free_transfers"] + 1)
            state["last_ft_accrual_gw"] = gw

        # Save checkpoint for this GW
        _ensure_checkpoint(state, gw, kb_meta)
        prev = next((e for e in state.get("log", []) if int(e.get("gw", -1)) == int(gw) - 1), None)
        if prev and prev.get("squad_codes"):
            prev_squad = set(map(int, prev["squad_codes"]))
            cp_key = str(int(gw))
            cp_squad = set(map(int, state["checkpoints"][cp_key]["squad"]))
            drift_add = list(cp_squad - prev_squad)
            drift_del = list(prev_squad - cp_squad)
            if drift_add or drift_del:
                # auto-restore baseline
                state["squad"] = sorted(prev_squad)
                state["checkpoints"][cp_key]["squad"] = state["squad"]

        dec = weekly_decision(
            players_df,
            st.session_state.full_kb,
            state,
            model_name,
            gw,
            extra_instructions=extra_instructions if gw == gw_now else None,
        )
        if dec.get("error"):
            break

        transfers = dec.get("transfers") or []
        ok, msg, new_bank, new_squad = _validate_transfers(
            players_df, state["squad"], state["bank"], transfers
        )
        before = set(map(int, state["checkpoints"][str(int(gw))]["squad"]))
        after  = set(map(int, new_squad))
        expected_changes = 2 * len(transfers)  # each swap adds +1 and removes +1
        if len(before ^ after) != expected_changes:
            # bail out (or log & skip committing this GW)
            break
        if not ok:
            break

        hit_cost = int(state.get("hit_cost", HIT_COST_DEFAULT))
        free_now = int(state.get("free_transfers", 0))
        t_count = len(transfers)
        points_hit = max(0, t_count - free_now) * hit_cost

        max_hit = int(state.get("max_hit", 1000))
        if points_hit > max_hit:
            break

        state["squad"] = new_squad
        state["bank"]  = float(new_bank)
        consumed_fts = min(t_count, free_now)
        state["free_transfers"] = max(0, free_now - consumed_fts)

        xi_codes = list(map(int, dec.get("xi_codes") or []))
        bench_codes = list(map(int, dec.get("bench_codes") or dec.get("bench_order") or []))
        ok, why = _validate_lineup(players_df, state["squad"], xi_codes, bench_codes)
        if not ok:
            break

        cap_code = int(dec.get("captain_code") or 0)
        if cap_code not in xi_codes:
            break

        chip = dec.get("chip", "NONE")
        if chip not in ("NONE", "TC", "BB"):
            chip = "NONE"
        if chip in ("TC", "BB") and not state["chips"].get(chip, False):
            chip = "NONE"

        pts = _compute_points(xi_codes, cap_code, bench_codes, gw, chip, code_to_id) - points_hit

        if chip in ("TC", "BB"):
            state["chips"][chip] = False

        # after you have new_squad, xi_codes, bench_codes, cap_code, etc.
        snap_after  = _snapshot(players_df, new_squad)
        snap_xi     = _snapshot(players_df, xi_codes)
        snap_bench  = _snapshot(players_df, bench_codes)
        
        entry = {
            "gw": int(gw),
            "made": bool(t_count > 0),
            "transfers": [{"out_code": int(t["out_code"]), "in_code": int(t["in_code"])} for t in transfers],
            "points_hit": int(points_hit),
            "chip": chip,
            "xi_codes": xi_codes,
            "bench_codes": bench_codes,
            "captain_code": cap_code,
            "points": int(pts),
            "bank": float(state["bank"]),
            "free_transfers": int(state["free_transfers"]),
            "squad_codes": list(map(int, state["squad"])),  # post-transfer 15
            "reason": dec.get("reason", ""),
            "bench_reason": dec.get("bench_reason", ""),
            "transfer_reasons": dec.get("transfer_reasons", []),
        
            # 🆕 snapshots for UI
            "snapshot_15": snap_after,     # full 15 after transfers
            "snapshot_xi": snap_xi,
            "snapshot_bench": snap_bench,
        }

        state["log"].append(entry)
        state["last_gw_processed"] = gw

        save_state(user_id, state)
        append_gw_log(user_id, gw, entry)

def rewind_and_regenerate_current_gw(
    user_id: str,
    kb_meta: dict,
    players_df: pd.DataFrame,
    model_name: str,
    extra_instructions: str | None = None
):
    """Set pointer back one and re-run a single GW (current), with optional user note."""
    if "auto_mgr" not in st.session_state:
        return False, "No state."
    state = st.session_state.auto_mgr
    gw_now = kb_meta.get("gw")
    if not gw_now:
        return False, "No current GW."
    if not state.get("squad"):
        return False, "No squad."

    # Restore pre-decision snapshot if allowed (deadline-aware)
    restored, why = _restore_checkpoint_if_allowed(state, int(gw_now), kb_meta)

    # Remove in-memory log for gw_now (DB history is immutable)
    state["log"] = [e for e in state["log"] if int(e.get("gw", -1)) != int(gw_now)]
    state["last_gw_processed"] = int(gw_now) - 1
    save_state(user_id, state)

    run_ai_auto_until_current(
        user_id=user_id,
        kb_meta=kb_meta,
        players_df=players_df,
        model_name=model_name,
        extra_instructions=extra_instructions,
    )
    return True, ("Regenerated." + (f" {why}" if restored else ""))


# ---------- refresh points (robust codes→id) ----------
def _best_code_to_id(
    players_df: pd.DataFrame | None,
    kb_meta: dict | None,
) -> Dict[int, int]:
    if kb_meta and kb_meta.get("code_to_id"):
        try:
            return {int(k): int(v) for k, v in kb_meta["code_to_id"].items()}
        except Exception:
            pass
    if players_df is not None and "code" in players_df.columns and "id" in players_df.columns:
        return {int(c): int(i) for c, i in zip(players_df["code"], players_df["id"])}
    return {}

def refresh_logged_points(
    user_id: str,
    players_df: pd.DataFrame | None = None,
    kb_meta: dict | None = None,
) -> int:
    """Recompute points for all logged GWs from official FPL history (works for legacy id logs too)."""
    if "auto_mgr" not in st.session_state:
        return 0
    state = st.session_state.auto_mgr

    code_to_id = _best_code_to_id(players_df, kb_meta or st.session_state.get("kb_meta"))
    id_to_code = {v: k for k, v in code_to_id.items()} if code_to_id else {}
    def ids_to_codes(lst):
        out = []
        for x in lst or []:
            try:
                xi = int(x)
                out.append(int(id_to_code.get(xi, xi)))
            except Exception:
                pass
        return out

    updated = 0
    for entry in state.get("log", []):
        gw = int(entry.get("gw", 0))
        xi_codes    = list(map(int, entry.get("xi_codes") or ids_to_codes(entry.get("xi_ids"))))
        bench_codes = list(map(int, entry.get("bench_codes") or ids_to_codes(entry.get("bench_ids") or entry.get("bench_order"))))
        cap_code    = entry.get("captain_code")
        if cap_code is None and entry.get("captain_id") is not None:
            cap_code = id_to_code.get(int(entry["captain_id"]), 0)
        cap_code = int(cap_code or 0)

        chip = (entry.get("chip") or "NONE").upper()
        new_pts = _compute_points(xi_codes, cap_code, bench_codes, gw, chip, code_to_id)

        if int(entry.get("points", -999999)) != int(new_pts):
            entry["points"] = int(new_pts)
            entry["xi_codes"] = xi_codes
            entry["bench_codes"] = bench_codes
            entry["captain_code"] = cap_code
            append_gw_log(user_id, gw, entry)
            updated += 1
    save_state(user_id, state)
    return updated

def force_redraft_gw1(
    user_id: str,
    players_df: pd.DataFrame,
    kb_text: str,
    model_name: str,
    extra_instructions: str | None = None,
) -> Tuple[bool, str]:
    """Re-draft a full legal 15 for GW1 using the LLM and replace state.squad (codes; no FT cost)."""
    if "auto_mgr" not in st.session_state:
        return False, "No state."
    state = st.session_state.auto_mgr
    budget = float(state.get("budget", 100.0))

    if not st.session_state.openai_key:
        return False, "no_api"

    obj = draft_initial_squad(
        players_df,
        kb_text,
        model_name,
        budget=budget,
        extra_instructions=extra_instructions,
        prior_squad_codes=state.get("squad") or None,
    )
    if obj.get("error"):
        return False, obj["error"]

    codes = obj.get("squad_codes") or []
    if (not codes) and obj.get("squad_ids") and "id" in players_df.columns:
        _, id_to_code = _ensure_maps(players_df)
        codes = [int(id_to_code.get(int(i), -1)) for i in obj["squad_ids"]]

    ok, why = _validate_initial(players_df, codes, budget)
    if not ok:
        return False, f"redraft_invalid:{why}"

    cost = float(players_df[players_df["code"].isin(codes)]["price"].sum())
    state["squad"] = list(map(int, codes))
    state["bank"]  = float(budget - cost)
    state.setdefault("free_transfers", 0)
    state.setdefault("chips", {"TC": True, "BB": True, "FH": True, "WC1": True, "WC2": True})
    state.setdefault("last_gw_processed", None)
    state.setdefault("last_ft_accrual_gw", 0)
    state.setdefault("budget", budget)
    state.setdefault("hit_cost", HIT_COST_DEFAULT)
    state.setdefault("max_hit", 0)
    state["seed_reason"] = obj.get("reason", "")

    save_state(user_id, state)
    return True, "Redrafted GW1 squad."
