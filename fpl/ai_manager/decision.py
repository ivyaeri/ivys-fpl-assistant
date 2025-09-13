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
from functools import lru_cache
import typing as _t

# ---------- utils ----------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _gw_deadline_utc(kb_meta: dict, gw: int) -> _t.Optional[datetime]:
    """
    Try to fetch the GW deadline (UTC) from kb_meta. If not present, return None.
    kb_meta may contain:
      - 'deadline_utc' for current GW (datetime or ISO string)
      - 'deadlines' or 'deadline_by_gw': {gw: ISO string or datetime}
    """
    if not isinstance(kb_meta, dict):
        return None
    # exact value for current gw
    if kb_meta.get("deadline_utc"):
        try:
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
    d = _gw_deadline_utc(kb_meta, gw)
    snap = {
        "gw": int(gw),
        "when_utc": _utc_now().isoformat(),
        "deadline_utc": d.isoformat() if d else None,
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
    """
    Robust JSON extractor:
    - Accepts raw JSON
    - Strips code fences if present
    - Falls back to the last balanced {...} block in the text
    """
    if not s:
        return {}
    s = s.strip()
    # strip code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M)
        s = s.strip()

    # try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # fallback: find last balanced JSON object
    last = None
    level = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if level == 0:
                start = i
            level += 1
        elif ch == "}":
            if level > 0:
                level -= 1
                if level == 0 and start != -1:
                    last = s[start:i+1]
    if last:
        try:
            return json.loads(last)
        except Exception:
            return {}
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
        code_to_id = {int(c): int(i) for c, i in zip(players_df["code"], players_df["id"])}
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
    """Return True if we changed the state (migrated from ids to codes)."""
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

# ---------- drafting helpers ----------

def _players_table_for_draft(players_df: pd.DataFrame) -> str:
    """
    Produce a compact CSV of candidate players to keep token usage sane for LLMs.
    Prioritize availability + form/ppg, then price.
    """
    cols = ["code","web_name","team_short","pos","price","form","status","selected_by","points_per_game"]
    df = players_df[[c for c in cols if c in players_df.columns]].copy()
    if df.empty:
        return ""
    out = []
    per_pos_head = {"GK": 20, "DEF": 50, "MID": 70, "FWD": 35}
    for pos, n in per_pos_head.items():
        sub = df[df["pos"] == pos].copy()
        if sub.empty:
            continue
        if "status" in sub:
            sub["__avail__"] = (sub["status"] == "a").astype(int)
            sort_cols = ["__avail__", "form", "points_per_game", "price"]
            ascending = [False, False, False, True]
        else:
            sort_cols = ["form", "points_per_game", "price"]
            ascending = [False, False, True]
        for c in ["form", "points_per_game", "price"]:
            if c not in sub.columns:
                sub[c] = 0.0
        sub = sub.sort_values(sort_cols, ascending=ascending).head(n)
        out.append(sub.drop(columns=[c for c in ["__avail__"] if c in sub.columns], errors="ignore"))
    tab = pd.concat(out, ignore_index=True) if out else df
    return tab.to_csv(index=False)

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

@lru_cache(maxsize=2048)
def _history_for_player(pid: int) -> list[dict]:
    """Cached FPL history fetch for speed and stability."""
    try:
        return fetch_player_history(int(pid)).get("history", [])
    except Exception:
        return []

def _event_points(code: int, gw: int, code_to_id: Dict[int,int]) -> int:
    try:
        pid = code_to_id.get(int(code))
        if not pid:
            return 0
        for g in _history_for_player(int(pid)):
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
    """
    Compatible ChatOpenAI init for newer and older langchain_openai versions.
    """
    try:
        # Newer versions
        return ChatOpenAI(model=model_name, api_key=st.session_state.openai_key, temperature=0.2)
    except TypeError:
        # Older versions
        return ChatOpenAI(model_name=model_name, openai_api_key=st.session_state.openai_key, temperature=0.2)

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
    table = _players_table_for_draft(players_df)

    sys = (
        "You are an elite Fantasy Premier League drafter. Always obey constraints and "
        "return STRICT JSON ONLY (no prose/markdown/code fences). Use player CODES."
    )

    prior_block = ""
    if prior_squad_codes:
        prior_sub = players_df[players_df["code"].isin([int(x) for x in prior_squad_codes])][
            ["code","web_name","team_short","pos","price","status","form","points_per_game"]
        ].sort_values(["pos","web_name"]).to_csv(index=False)
        prior_block = f"""
PRIOR_SQUAD_CODES: {list(map(int, prior_squad_codes))}
PRIOR_SQUAD_TABLE (CSV):
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
PLAYERS (CSV: code, name, team, pos, price, form, status, selected_by, ppg):
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
- Total price ≤ budget; exact 2/5/5/3 shape; ≤3 per club; codes must be from the PLAYERS list.
- If PRIOR_SQUAD_CODES are given, keep changes minimal unless instructions mandate otherwise.
"""

    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    obj = _json_from_text(raw)
    if not obj:
        # retry once with a shorter user block
        raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr[:8000]}]).content
        obj = _json_from_text(raw)
    if not obj:
        return {"error":"parse","raw": raw[:2000]}

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
    table = sub.to_csv(index=False)
    chips = [k for k,v in state.get("chips",{}).items() if v] or ["NONE"]

    note = (extra_instructions or "").strip()
    if note:
        note = note[:800]

    sys = (
        "You are an autonomous FPL manager. "
        "Return STRICT JSON only. No markdown, no comments. Use player CODES."
    )
    usr = f"""
FPL MASTER STRATEGIST - GW{gw} OPTIMAL DECISION ENGINE

You are an elite FPL manager with deep statistical knowledge. Your goal: MAXIMIZE POINTS while building long-term value.

══════════ CURRENT STATE ══════════
Squad: {table}
Bank: £{state['bank']:.1f}m | Free Transfers: {state['free_transfers']} | Chips: {chips}

══════════ CRITICAL SUCCESS FACTORS ══════════

**POINTS MAXIMIZATION HIERARCHY:**
1. **Starting XI Strength** - Your 11 must be point machines
2. **Budget Optimization** - NEVER waste money in bank (max 1.0m buffer)
3. **Fixture Leverage** - Target easiest opponents & attacking situations
4. **Form Over Fame** - In-form £6m > Out-of-form £10m
5. **Template Balance** - Mix of safe picks + differentials for rank moves

**MANDATORY ANALYSIS CHECKLIST:**

**FIXTURE INTELLIGENCE (Next 3-5 GWs):**
- FDR ratings: 1-2 (green) = prioritize, 4-5 (red) = avoid
- Home advantage vs defensive strength
- Goals conceded trends (attack targets)
- Clean sheet probability (defensive targets)

**FORM & EXPECTED PERFORMANCE:**
- Points last 3 GWs vs season average
- xG/xA vs actual (over/underperformance)
- Minutes played % (rotation risk)
- Bonus point magnets (shots, key passes, defensive actions)

**VALUE ENGINEERING:**
- Points per million spent efficiency
- Budget enablers performing above price point
- Premium assets justifying cost vs cheaper alternatives
- Price change momentum (rising/falling)

**OWNERSHIP PSYCHOLOGY:**
- Template players (safe but limited upside)
- Differentials with explosive potential (5-15% owned)
- Captaincy options beyond obvious choices
- Anti-template opportunities (fading popular picks in bad fixtures)

**POSITIONAL OPTIMIZATION:**
- GK: Fixtures + save potential + penalty save history
- DEF: Clean sheets + attacking returns + BPS potential  
- MID: Goals/assists + set pieces + advanced positions
- FWD: Shot volume + big chances + supporting cast quality

**BUDGET MAXIMIZATION RULES:**
- Bank should be £0.0-1.0m (emergency buffer only)
- Every unused million = ~2-3 points lost per GW
- Upgrade path: Identify weakest starter, find best replacement in budget
- Value traps: Expensive benchwarmer vs playing budget option

**STRATEGIC THINKING:**

**Template Disruption:**
- When template fails: Be contrarian on popular picks with bad fixtures
- Differential captaincy: Target 5-20% owned in good spots
- Rank climbing: Take calculated risks when behind

**Risk Management:**
- Rotation prone players before busy periods
- Injury concerns from recent games/internationals
- New signings adaptation period
- Tactical changes affecting player roles

**Long-term Setup:**
- Building toward DGWs (2-3 GWs ahead)
- Wildcard timing optimization
- Team value growth through price rises

══════════ DECISION FRAMEWORK ══════════

**TRANSFER PRIORITY MATRIX:**
1. Remove: Injured/suspended/not playing
2. Remove: Playing but terrible fixtures (FDR 4-5)
3. Remove: Expensive underperformers vs budget alternatives  
4. Add: In-form players with great fixtures
5. Add: Budget gems outperforming price point
6. Add: Differentials with explosive upside

**FORMATION SELECTION:**
Choose based on your best 11 players, not rigid structures:
- 3-5-2: Strong midfield, weak forward depth
- 4-4-2: Balanced, two strong forwards
- 3-4-3: Premium forward heavy
- 5-4-1: Defensive fixtures, one premium forward

**CAPTAINCY ALGORITHM:**
1. Highest ceiling (goals/assists potential)
2. Best fixture (FDR 1-2, home advantage)
3. Current form (points last 3 GWs)
4. Ownership consideration (template vs differential)

KNOWLEDGE BASE:
{kb_text}

MANAGER DIRECTIVE:
{note if note else f"Maximize GW{gw} points while optimizing budget usage"}

══════════ ELITE OUTPUT REQUIRED ══════════
STRICT OUTPUT CONTRACT — READ CAREFULLY

- Output MUST be a single valid JSON object.
- No markdown, code fences, comments, or extra prose before/after the JSON.
- Use player CODES only (from the Squad table above).
- All keys below are REQUIRED; types must be exact (numbers as numbers, not strings).
- xi_codes + bench_codes must be disjoint and together cover all 15 current squad codes.
- Exactly 1 GK in xi_codes; formation must be valid (e.g., 3-4-3, 3-5-2, 4-4-2, 4-5-1, 5-4-1).
- captain_code and vice_captain_code must be different and both inside xi_codes.
- Every transfer must be like-for-like by position, ≤3 per club after application, and within budget.
- final_bank must equal starting bank ± price deltas from transfers (≥ 0.0), and keep ≤ £1.0m unless strongly justified.
- transfer_reasons must be a list of strings in the same order as transfers.
- If making no transfers, return "transfers": [] and set final_bank to the current bank.

Return EXACTLY this JSON shape (all keys required):

{{
  "schema_version": "v2",

  "chip": "NONE" | "TC" | "BB",
  "transfers": [{{"out_code": <int>, "in_code": <int>}}],

  "xi_codes": [<11 ints>],
  "bench_codes": [<4 ints>],
  "captain_code": <int>,
  "vice_captain_code": <int>,

  "reason": "<2-3 sentence high-level approach>",
  "bench_reason": "<bench ordering logic>",
  "transfer_reasons": ["<one-line rationale per transfer>"],

  "strategy_summary": "<same as reason or expanded>",
  "budget_optimization": "<how you maximized the £{state['bank']:.1f}m + any sale proceeds>",
  "fixture_leverage": "<key fixtures/players you're targeting this GW>",
  "form_rationale": "<which in-form players you prioritized and why>",
  "differentials": ["<any differential picks 5-20% owned with reasoning>"],
  "template_stance": "<are you following template or contrarian this GW and why>",

  "transfer_breakdown": [
    {{
      "out": "<player name>",
      "in": "<player name>",
      "out_code": <int>,
      "in_code": <int>,
      "cost_impact": "<+/-£X.Ym>",
      "reason": "<fixture/form/value justification>",
      "risk_level": "LOW" | "MEDIUM" | "HIGH"
    }}
  ],

  "xi_justification": "<why these 11 over alternatives>",
  "captain_logic": "<ceiling vs safety vs differential reasoning>",
  "bench_strategy": "<same as bench_reason or elaboration>",
  "key_risks": "<biggest threats to this strategy>",
  "next_gw_setup": "<how this prepares you for GW{gw+1}>",

  "final_bank": <float>,
  "budget_efficiency_score": <int>
}}

══════════ VALIDATION CHECKLIST (apply before you print JSON) ══════════
- JSON only; no extra text.
- All required keys present and non-null.
- xi_codes has 11, bench_codes has 4; union = 15; intersection = ∅.
- Exactly 1 GK in xi_codes; valid DEF-MID-FWD counts (e.g., 3-4-3 / 3-5-2 / 4-4-2 / 4-5-1 / 5-4-1).
- captain_code and vice_captain_code ∈ xi_codes and are different.
- Each transfer is like-for-like by position; resulting squad obeys ≤3 per club and budget.
- transfer_reasons aligns one-to-one (same order) with transfers.
- final_bank computed from listed transfers using prices in the Squad table; ≥ 0.0.
- budget_efficiency_score is an integer 1–10 (≥ 8 if you keep >£1.0m, justify in budget_optimization).
"""
    raw = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}]).content
    obj = _json_from_text(raw)
    if not obj:
        return {"error":"parse","raw": raw[:2000]}

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
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] No OpenAI key; stopping.")
            break

        if gw > 1 and state.get("last_ft_accrual_gw") != gw:
            # FPL caps free transfers at 5
            state["free_transfers"] = min(5, state["free_transfers"] + 1)
            state["last_ft_accrual_gw"] = gw

        # Save checkpoint for this GW
        _ensure_checkpoint(state, gw, kb_meta)

        # Guard against manual drift since previous GW
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

        kb_text = (getattr(st.session_state, "full_kb", None)
                   or (kb_meta or {}).get("kb_text")
                   or "")

        dec = weekly_decision(
            players_df,
            kb_text,
            state,
            model_name,
            gw,
            extra_instructions=extra_instructions if gw == gw_now else None,
        )
        if dec.get("error"):
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] Decision error: {dec.get('error')}")
            break

        transfers = dec.get("transfers") or []
        ok, msg, new_bank, new_squad = _validate_transfers(
            players_df, state["squad"], state["bank"], transfers
        )

        before = set(map(int, state["checkpoints"][str(int(gw))]["squad"]))
        after  = set(map(int, new_squad))
        expected_changes = 2 * len(transfers)  # each swap adds +1 and removes +1

        if len(before ^ after) != expected_changes:
            # proceed anyway; just log it for debugging
            st.session_state.setdefault("ai_mgr_logs", []).append(
                f"[GW{gw}] Unexpected change-count: expected {expected_changes}, got {len(before ^ after)}"
            )
        if not ok:
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] Transfer validation failed: {msg}")
            break

        hit_cost = int(state.get("hit_cost", HIT_COST_DEFAULT))
        free_now = int(state.get("free_transfers", 0))
        t_count = len(transfers)
        points_hit = max(0, t_count - free_now) * hit_cost

        max_hit = int(state.get("max_hit", 1000))
        if points_hit > max_hit:
            st.session_state.setdefault("ai_mgr_logs", []).append(
                f"[GW{gw}] Hit {points_hit} exceeds max_hit {max_hit}; aborting."
            )
            break

        state["squad"] = new_squad
        state["bank"]  = float(new_bank)
        consumed_fts = min(t_count, free_now)
        state["free_transfers"] = max(0, free_now - consumed_fts)

        xi_codes = list(map(int, dec.get("xi_codes") or []))
        bench_codes = list(map(int, dec.get("bench_codes") or dec.get("bench_order") or []))
        ok, why = _validate_lineup(players_df, state["squad"], xi_codes, bench_codes)
        if not ok:
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] Lineup invalid: {why}")
            break

        cap_code  = int(dec.get("captain_code") or 0)
        vice_code = int(dec.get("vice_captain_code") or 0)
        if cap_code not in xi_codes:
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] Captain not in XI; aborting.")
            break
        if vice_code and vice_code not in xi_codes:
            st.session_state.setdefault("ai_mgr_logs", []).append(f"[GW{gw}] Vice not in XI; continuing, but check prompt.")

        chip = (dec.get("chip") or "NONE").upper()
        if chip not in ("NONE", "TC", "BB"):
            chip = "NONE"
        if chip in ("TC", "BB") and not state["chips"].get(chip, False):
            chip = "NONE"

        pts = _compute_points(xi_codes, cap_code, bench_codes, gw, chip, code_to_id) - points_hit

        if chip in ("TC", "BB"):
            state["chips"][chip] = False

        # snapshots for UI
        snap_after  = _snapshot(players_df, new_squad)
        snap_xi     = _snapshot(players_df, xi_codes)
        snap_bench  = _snapshot(players_df, bench_codes)

        # ---- Gather ALL JSON fields the model returned ----
        # legacy/UI keys with aliases
        reason = dec.get("reason") or dec.get("strategy_summary") or ""
        bench_reason = dec.get("bench_reason") or dec.get("bench_strategy") or ""

        # normalize transfer_reasons -> list[str]
        transfer_reasons = dec.get("transfer_reasons")
        if not (isinstance(transfer_reasons, list) and all(isinstance(x, str) for x in transfer_reasons or [])):
            transfer_reasons = []
            for x in dec.get("transfer_breakdown") or []:
                if isinstance(x, dict):
                    outn = x.get("out") or x.get("out_name") or x.get("out_code") or "?"
                    inn  = x.get("in")  or x.get("in_name")  or x.get("in_code")  or "?"
                    why  = x.get("reason") or x.get("rationale") or ""
                    ri   = x.get("risk_level") or x.get("risk") or ""
                    s = f"{outn} → {inn}"
                    if why: s += f": {why}"
                    if ri:  s += f" (risk {ri})"
                    transfer_reasons.append(s)
                else:
                    transfer_reasons.append(str(x))

        # rich analysis fields
        strategy_summary        = dec.get("strategy_summary")
        budget_optimization     = dec.get("budget_optimization")
        fixture_leverage        = dec.get("fixture_leverage")
        form_rationale          = dec.get("form_rationale")
        differentials           = dec.get("differentials")
        template_stance         = dec.get("template_stance")
        transfer_breakdown_raw  = dec.get("transfer_breakdown") or []
        xi_justification        = dec.get("xi_justification")
        captain_logic           = dec.get("captain_logic")
        bench_strategy          = dec.get("bench_strategy")
        key_risks               = dec.get("key_risks")
        next_gw_setup           = dec.get("next_gw_setup")
        model_final_bank        = dec.get("final_bank")
        budget_efficiency_score = dec.get("budget_efficiency_score")

        # coerce model_final_bank if possible (store both declared & computed)
        try:
            model_final_bank = float(model_final_bank)
        except Exception:
            model_final_bank = None

        entry = {
            "schema_version": "v2",
            "gw": int(gw),
            "made": bool(t_count > 0),
            "transfers": [{"out_code": int(t["out_code"]), "in_code": int(t["in_code"])} for t in transfers],
            "points_hit": int(points_hit),
            "chip": chip,
            "xi_codes": xi_codes,
            "bench_codes": bench_codes,
            "captain_code": cap_code,
            "vice_captain_code": vice_code,
            "points": int(pts),
            "bank": float(state["bank"]),  # bank after applying transfers
            "free_transfers": int(state["free_transfers"]),
            "squad_codes": list(map(int, state["squad"])),  # post-transfer 15

            # legacy + normalized
            "reason": reason,
            "bench_reason": bench_reason,
            "transfer_reasons": transfer_reasons,

            # full analysis payload from model
            "strategy_summary": strategy_summary,
            "budget_optimization": budget_optimization,
            "fixture_leverage": fixture_leverage,
            "form_rationale": form_rationale,
            "differentials": differentials,
            "template_stance": template_stance,
            "transfer_breakdown": transfer_breakdown_raw,
            "xi_justification": xi_justification,
            "captain_logic": captain_logic,
            "bench_strategy": bench_strategy,
            "key_risks": key_risks,
            "next_gw_setup": next_gw_setup,
            "final_bank_model": model_final_bank,
            "budget_efficiency_score": budget_efficiency_score,

            # snapshots for UI
            "snapshot_15": snap_after,
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
    """
    Recompute points for all logged GWs using historical per-GW data
    (stored in st.session_state["gw_history"]). 
    Once a GW is locked, don't overwrite it again.
    """
    if "auto_mgr" not in st.session_state:
        return 0
    state = st.session_state.auto_mgr

    # Historical GW datasets
    gw_history: dict[int, pd.DataFrame] = st.session_state.get("gw_history", {})

    updated = 0
    for entry in state.get("log", []):
        gw = int(entry.get("gw", 0))
        if not gw:
            continue

        # Skip if locked already
        if entry.get("points_locked", False):
            continue

        # Pick dataset for this GW
        gw_players = gw_history.get(gw) or players_df
        if gw_players is None:
            continue

        # Build code→id mapping for that GW
        code_to_id = _best_code_to_id(gw_players, kb_meta or st.session_state.get("kb_meta"))
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

        xi_codes    = list(map(int, entry.get("xi_codes") or ids_to_codes(entry.get("xi_ids"))))
        bench_codes = list(map(int, entry.get("bench_codes") or ids_to_codes(entry.get("bench_ids") or entry.get("bench_order"))))
        cap_code    = entry.get("captain_code")
        if cap_code is None and entry.get("captain_id") is not None:
            cap_code = id_to_code.get(int(entry["captain_id"]), 0)
        cap_code = int(cap_code or 0)

        chip = (entry.get("chip") or "NONE").upper()
        points_hit = int(entry.get("points_hit", 0))

        # Compute points from correct GW dataset
        new_pts = _compute_points(xi_codes, cap_code, bench_codes, gw, chip, code_to_id) - points_hit

        # Update and lock
        if int(entry.get("points", -999999)) != int(new_pts):
            entry["points"] = int(new_pts)
            entry["xi_codes"] = xi_codes
            entry["bench_codes"] = bench_codes
            entry["captain_code"] = cap_code
            entry["points_locked"] = True
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
