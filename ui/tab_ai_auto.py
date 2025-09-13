# ui/tab_ai_auto.py
import streamlit as st
import pandas as pd

from config import MODEL_NAME
from fpl.ai_manager.decision import (
    ensure_initial_squad_with_ai,
    rewind_and_regenerate_current_gw,
    run_ai_auto_until_current,
    refresh_logged_points,
    force_redraft_gw1,
)

# ---------- helpers ----------
def _maps(players_df: pd.DataFrame):
    """Return (code->id, id->code). If columns are missing, return empties."""
    if "code" in players_df.columns and "id" in players_df.columns:
        c2i = {int(c): int(i) for c, i in zip(players_df["code"], players_df["id"])}
        i2c = {int(i): int(c) for i, c in zip(players_df["id"], players_df["code"])}
        return c2i, i2c
    return {}, {}

def _pname_by_code(players_df: pd.DataFrame, code) -> str:
    if code is None and code != 0:
        return "—"
    if "code" not in players_df.columns:
        return f"CODE {code}"
    try:
        code = int(code)
    except Exception:
        return str(code)
    row = players_df.loc[players_df["code"] == code]
    return row["web_name"].iloc[0] if not row.empty else f"CODE {code}"

def _pname_by_id(players_df: pd.DataFrame, pid) -> str:
    if pid is None and pid != 0:
        return "—"
    if "id" not in players_df.columns:
        return f"ID {pid}"
    try:
        pid = int(pid)
    except Exception:
        return str(pid)
    row = players_df.loc[players_df["id"] == pid]
    return row["web_name"].iloc[0] if not row.empty else f"ID {pid}"

def _snapshot_df(players_df: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    """Small table for the UI based on codes."""
    if "code" not in players_df.columns:
        return pd.DataFrame()
    try:
        codes = [int(c) for c in (codes or [])]
    except Exception:
        return pd.DataFrame()
    sub = players_df[players_df["code"].isin(codes)].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [
        "web_name","team_short","pos","price","form","status","selected_by","points_per_game","code"
    ]
    return sub[[c for c in cols if c in sub.columns]]

def _df_from_snapshot_list(snapshot_list: list[dict]) -> pd.DataFrame:
    """Turn the decision snapshots into a DataFrame, preserving useful cols."""
    if not snapshot_list:
        return pd.DataFrame()
    df = pd.DataFrame(snapshot_list)
    wanted = ["web_name","team_short","pos","price","form","status","points_per_game","code"]
    keep = [c for c in wanted if c in df.columns]
    return df[keep] if keep else df

def _canonicalize_entry(entry: dict, i2c: dict) -> dict:
    """
    Normalize an entry into a single canonical schema, regardless of old/new JSONs.
    Produces keys we render: transfers(list[{out_code,in_code}]), xi_codes, bench_codes,
    captain_code, vice_captain_code, squad_codes, reason, bench_reason,
    transfer_reasons(list[str]), chip, points_hit, points, bank, free_transfers,
    plus snapshots and full analysis fields if present.
    """
    e = dict(entry or {})

    # --- id->code helper
    def _ids_to_codes(lst):
        out = []
        for x in lst or []:
            try:
                xi = int(x)
                out.append(int(i2c.get(xi, xi)))
            except Exception:
                pass
        return out

    # 1) transfers -> list[{out_code,in_code}]
    moves = e.get("transfers") or e.get("moves") or []
    if isinstance(moves, dict):  # very old single-transfer
        moves = [moves]
    transfers = []
    for mv in (moves or []):
        if not isinstance(mv, dict):
            continue
        oc = mv.get("out_code")
        ic = mv.get("in_code")
        # legacy id keys
        if oc is None and mv.get("out_id") is not None:
            oc = i2c.get(int(mv["out_id"]), mv["out_id"])
        if ic is None and mv.get("in_id") is not None:
            ic = i2c.get(int(mv["in_id"]), mv["in_id"])
        # very old ambiguous "out"/"in" (ids)
        if oc is None and mv.get("out") is not None:
            try: oc = i2c.get(int(mv["out"]), mv["out"])
            except Exception: oc = mv["out"]
        if ic is None and mv.get("in") is not None:
            try: ic = i2c.get(int(mv["in"]), mv["in"])
            except Exception: ic = mv["in"]
        try:
            transfers.append({"out_code": int(oc), "in_code": int(ic)})
        except Exception:
            pass
    e["transfers"] = transfers

    # 2) xi/bench to codes
    xi_codes = e.get("xi_codes")
    if not xi_codes and e.get("xi_ids"):
        xi_codes = _ids_to_codes(e.get("xi_ids"))
    bench_codes = e.get("bench_codes")
    if not bench_codes:
        bench_legacy = e.get("bench_ids") or e.get("bench_order") or []
        bench_codes = _ids_to_codes(bench_legacy)
    e["xi_codes"] = [int(x) for x in (xi_codes or [])]
    e["bench_codes"] = [int(x) for x in (bench_codes or [])]

    # 3) captain/vice to codes
    cap = e.get("captain_code")
    if cap is None and e.get("captain_id") is not None:
        cap = i2c.get(int(e["captain_id"]), e["captain_id"])
    vice = e.get("vice_captain_code")
    if vice is None and e.get("vice_captain_id") is not None:
        vice = i2c.get(int(e["vice_captain_id"]), e["vice_captain_id"])
    try:
        e["captain_code"] = int(cap or 0)
    except Exception:
        e["captain_code"] = 0
    try:
        e["vice_captain_code"] = int(vice or 0)
    except Exception:
        e["vice_captain_code"] = 0

    # 4) squad to codes
    sc = e.get("squad_codes")
    if not sc and e.get("squad_ids"):
        sc = _ids_to_codes(e.get("squad_ids"))
    e["squad_codes"] = [int(x) for x in (sc or [])]

    # 5) reasons / breakdown normalization
    reason = e.get("reason") or e.get("strategy_summary") or ""
    bench_reason = e.get("bench_reason") or e.get("bench_strategy") or ""
    tbreak = e.get("transfer_breakdown") or []
    treasons = e.get("transfer_reasons")
    if isinstance(treasons, list) and all(isinstance(x, str) for x in treasons):
        transfer_reasons = treasons
    else:
        transfer_reasons = []
        for x in (tbreak if isinstance(tbreak, list) else []):
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
    e["reason"] = reason
    e["bench_reason"] = bench_reason
    e["transfer_reasons"] = transfer_reasons

    # 6) chip normalization
    chip = (e.get("chip") or "NONE").upper()
    e["chip"] = chip if chip in ("NONE", "TC", "BB", "FH", "WC1", "WC2") else "NONE"

    # 7) numeric hygiene
    try: e["points_hit"] = int(e.get("points_hit", 0))
    except Exception: e["points_hit"] = 0
    try: e["points"] = int(e.get("points", 0))
    except Exception: pass
    try: e["free_transfers"] = int(e.get("free_transfers", 0))
    except Exception: pass
    try: e["bank"] = float(e.get("bank", 0.0))
    except Exception: pass
    try:
        if e.get("final_bank_model") is not None:
            e["final_bank_model"] = float(e.get("final_bank_model"))
        elif e.get("final_bank") is not None:
            e["final_bank_model"] = float(e.get("final_bank"))
    except Exception:
        pass

    # copy through analysis fields as-is (if present)
    for k in [
        "schema_version","strategy_summary","budget_optimization","fixture_leverage",
        "form_rationale","differentials","template_stance","transfer_breakdown",
        "xi_justification","captain_logic","bench_strategy","key_risks",
        "next_gw_setup","budget_efficiency_score"
    ]:
        if k in e:
            e[k] = e[k]

    # keep snapshots if present
    for k in ["snapshot_15","snapshot_xi","snapshot_bench"]:
        if k in e:
            e[k] = e[k]

    return e

# ---------- main ----------
def render_ai_tab(players_df: pd.DataFrame, kb_meta: dict, user_id: str):
    st.subheader("🧠 AI Auto Manager — LLM-only")

    if "auto_mgr" not in st.session_state:
        st.info("State not loaded yet.")
        return

    state = st.session_state.auto_mgr
    c2i, i2c = _maps(players_df)

    # ---------- No squad yet: offer AI GW1 draft, then process ----------
    if not state.get("squad"):
        col1, col2 = st.columns([1, 3])

        with col1:
            disabled = not bool(st.session_state.openai_key)
            if st.button("🧠 Draft GW1 Squad (AI)", disabled=disabled):
                with st.spinner("Asking the model to draft your 15..."):
                    ensure_initial_squad_with_ai(
                        user_id=user_id,
                        players_df=players_df,
                        kb_text=st.session_state.get("full_kb", ""),
                        model_name=MODEL_NAME,
                        budget=100.0,
                    )

                squad_codes = st.session_state.get("auto_mgr", {}).get("squad") or []
                if len(squad_codes) == 15:
                    with st.spinner("Locking in GW decisions…"):
                        run_ai_auto_until_current(
                            user_id=user_id
