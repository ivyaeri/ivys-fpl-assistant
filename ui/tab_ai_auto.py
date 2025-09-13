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

    # copy through analysis fields if present
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
                            user_id=user_id,
                            kb_meta=kb_meta,
                            players_df=players_df,
                            model_name=MODEL_NAME,
                            extra_instructions=None,
                        )
                    st.success("Drafted and processed the current GW. See the log below.")
                    st.rerun()
                else:
                    reason = st.session_state.get("auto_mgr", {}).get("seed_origin", "unknown")
                    st.error(f"Draft failed ({reason}). Check your API key and try again.")

        with col2:
            if not st.session_state.openai_key:
                st.info("Add your OpenAI API key in the sidebar, then click **Draft GW1 Squad (AI)**.")
            else:
                st.info("The model will pick a legal 15 (2 GK / 5 DEF / 5 MID / 3 FWD, ≤3/club, ≤£100m).")
        return

    # ---------- With a squad: controls & regenerate ----------
    gw_now = int(kb_meta.get("gw") or 0)

    with st.expander("Optional: add instructions / redraft controls", expanded=False):
        st.caption("Examples: “prefer Arsenal defenders”, “avoid flagged players”, “consider BB if bench is strong”.")
        user_note = st.text_area(
            "Manager instructions (optional)",
            value="",
            height=90,
            placeholder="Type any constraints or preferences…",
        )
        force_redraft_toggle = (gw_now == 1)

    colA, colB = st.columns([1, 3])
    with colA:
        regen_disabled = not bool(st.session_state.openai_key)
        if st.button("🔁 Regenerate this GW (AI)", type="primary", disabled=regen_disabled):
            with st.spinner("Re-evaluating this gameweek…"):
                if gw_now == 1 and force_redraft_toggle:
                    ok, msg = force_redraft_gw1(
                        user_id=user_id,
                        players_df=players_df,
                        kb_text=st.session_state.get("full_kb", ""),
                        model_name=MODEL_NAME,
                        extra_instructions=(user_note or None),
                    )
                    if not ok:
                        st.error(f"Redraft failed: {msg}")
                        st.stop()

                ok, msg = rewind_and_regenerate_current_gw(
                    user_id=user_id,
                    kb_meta=kb_meta,
                    players_df=players_df,
                    model_name=MODEL_NAME,
                    extra_instructions=(user_note or None),
                )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)
    with colB:
        st.caption(f"User: **{user_id}**  ·  Current GW: **{gw_now or '—'}**")

    # ---------- Maintenance ----------
    with st.expander("Maintenance", expanded=False):
        st.caption("Recompute points for all logged GWs from official FPL history (useful after a GW finishes).")
        if st.button("↻ Refresh points for finished GWs"):
            n = refresh_logged_points(user_id, players_df=players_df, kb_meta=kb_meta)
            st.success(f"Updated {n} gameweek(s).")
            st.rerun()

    # ---------- Debug logs from backend ----------
    if st.session_state.get("ai_mgr_logs"):
        with st.expander("Engine notes", expanded=False):
            for line in st.session_state.get("ai_mgr_logs", []):
                st.text(line)

    # ---------- Weekly logs ----------
    logs = state.get("log") or []
    if not logs:
        st.info("No gameweeks processed yet.")
        return

    for raw_entry in sorted(logs, key=lambda x: x.get("gw", 0), reverse=True):
        entry = _canonicalize_entry(raw_entry, i2c)

        header_bits = [
            f"GW {entry.get('gw', '?')}",
            f"Points: {entry.get('points', 0)}",
            f"Bank £{float(entry.get('bank', 0.0)):.1f}",
            f"FTs {int(entry.get('free_transfers', 0))}",
            f"Hits {int(entry.get('points_hit', 0))}",
        ]
        chip = (entry.get("chip") or "NONE").upper()
        if chip != "NONE":
            header_bits.append(f"Chip {chip}")
        if entry.get("schema_version"):
            header_bits.append(f"Schema {entry['schema_version']}")
        if entry.get("redraft"):
            header_bits.append("Full redraft")

        with st.expander(" — ".join(header_bits), expanded=(entry.get("gw") == gw_now)):
            # ----- Metrics row (Captain + Vice added)
            cap_name  = _pname_by_code(players_df, entry.get("captain_code")) if entry.get("captain_code") else "—"
            vice_name = _pname_by_code(players_df, entry.get("vice_captain_code")) if entry.get("vice_captain_code") else "—"

            cols_meta = st.columns(5)
            cols_meta[0].metric("Points", entry.get("points", 0))
            cols_meta[1].metric("Points hit", entry.get("points_hit", 0))
            cols_meta[2].metric("Chip", chip)
            cols_meta[3].metric("Captain", cap_name)
            cols_meta[4].metric("Vice captain", vice_name)

            # ----- Bank summary (engine vs model)
            mfb = entry.get("final_bank_model", None)
            cols_bank = st.columns(2)
            cols_bank[0].markdown(f"**Engine bank (post-transfers):** £{float(entry.get('bank', 0.0)):.1f}m")
            if mfb is not None:
                cols_bank[1].markdown(f"**Model declared final_bank:** £{float(mfb):.1f}m")
                try:
                    diff = float(entry.get("bank", 0.0)) - float(mfb)
                    if abs(diff) > 0.05:
                        st.warning(f"Bank mismatch vs model: {diff:+.2f}m (engine − model)")
                except Exception:
                    pass

            # ----- Transfers (canonical codes)
            moves = entry.get("transfers") or []
            if not moves:
                st.markdown("**No transfer made.**")
            else:
                for mv in moves:
                    st.markdown(
                        f"**Transfer:** {_pname_by_code(players_df, mv.get('out_code'))} → {_pname_by_code(players_df, mv.get('in_code'))}"
                    )

            # ----- Transfer breakdown table (if present)
            tbreak = entry.get("transfer_breakdown") or []
            if isinstance(tbreak, list) and tbreak:
                df_tb = pd.DataFrame(tbreak)
                prefer_cols = ["out","out_code","in","in_code","cost_impact","reason","risk_level"]
                use_cols = [c for c in prefer_cols if c in df_tb.columns]
                df_tb = df_tb[use_cols] if use_cols else df_tb
                st.markdown("**Transfer breakdown (AI):**")
                st.dataframe(df_tb, use_container_width=True)

            # ----- Reasons & analysis
            st.markdown(f"**Reason (AI):** {entry.get('reason','')}")
            st.markdown(f"**Bench Reason (AI):** {entry.get('bench_reason','')}")
            treasons = entry.get("transfer_reasons") or []
            if isinstance(treasons, list) and treasons:
                st.caption("Transfer notes: " + "; ".join(map(str, treasons)))
            elif isinstance(treasons, str) and treasons.strip():
                st.caption(f"Transfer notes: {treasons}")

            cols_top = st.columns(2)
            with cols_top[0]:
                if entry.get("strategy_summary"):
                    st.markdown(f"**Strategy:** {entry['strategy_summary']}")
                if entry.get("fixture_leverage"):
                    st.markdown(f"**Fixtures leveraged:** {entry['fixture_leverage']}")
                if entry.get("template_stance"):
                    st.caption(f"Template stance: {entry['template_stance']}")
                if entry.get("xi_justification"):
                    st.caption(f"XI justification: {entry['xi_justification']}")
            with cols_top[1]:
                if entry.get("budget_optimization"):
                    st.markdown(f"**Budget:** {entry['budget_optimization']}")
                if entry.get("form_rationale"):
                    st.markdown(f"**Form rationale:** {entry['form_rationale']}")
                if entry.get("budget_efficiency_score") is not None:
                    st.caption(f"Budget efficiency: {entry['budget_efficiency_score']}/10")

            if entry.get("differentials"):
                diffs = entry["differentials"]
                if isinstance(diffs, list):
                    st.caption("Differentials: " + ", ".join(map(str, diffs)))
                else:
                    st.caption(f"Differentials: {diffs}")
            if entry.get("key_risks"):
                st.caption(f"Key risks: {entry['key_risks']}")
            if entry.get("next_gw_setup"):
                st.caption(f"Next GW setup: {entry['next_gw_setup']}")
            if entry.get("captain_logic"):
                st.caption(f"Captain logic: {entry['captain_logic']}")

            # ----- XI/Bench/Squad (prefer embedded snapshots)
            xi_codes    = entry.get("xi_codes") or []
            bench_codes = entry.get("bench_codes") or []
            cap_code    = int(entry.get("captain_code") or 0)
            vice_code   = int(entry.get("vice_captain_code") or 0)

            # NEW: Bench order (names) line
            if bench_codes:
                bench_names = [_pname_by_code(players_df, c) for c in bench_codes]
                st.caption("Bench order: " + " → ".join(map(str, bench_names)))

            week = _df_from_snapshot_list(entry.get("snapshot_15") or [])
            if week.empty:
                week = _snapshot_df(players_df, entry.get("squad_codes") or [])
            if not week.empty:
                xi_set = set(int(x) for x in xi_codes)
                bench_set = set(int(x) for x in bench_codes)

                week["XI"] = week["code"].apply(lambda c: "Yes" if int(c) in xi_set else "")
                week["Bench"] = week["code"].apply(lambda c: "Yes" if int(c) in bench_set else "")
                week["Captain"] = week["code"].apply(lambda c: "C" if int(c) == cap_code else "")
                if vice_code:
                    week["Vice"] = week["code"].apply(lambda c: "V" if int(c) == vice_code else "")

                display_cols = [
                    "web_name","team_short","pos","price","form","status","points_per_game",
                    "XI","Bench","Captain"
                ]
                if "Vice" in week.columns:
                    display_cols.append("Vice")

                # sort Captain desc, Vice desc (if present), XI desc, then pos/name
                sort_cols = ["Captain"]
                ascending = [False]
                if "Vice" in week.columns:
                    sort_cols.append("Vice")
                    ascending.append(False)
                sort_cols += ["XI","pos","web_name"]
                ascending += [False, True, True]

                week = week[[c for c in display_cols if c in week.columns]] \
                    .sort_values(sort_cols, ascending=ascending)

                st.markdown("**Full 15-man squad (this GW):**")
                st.dataframe(week, use_container_width=True)
                st.markdown(f"**Captain:** {_pname_by_code(players_df, cap_code) if cap_code else '—'}")
                if vice_code:
                    st.caption(f"Vice: {_pname_by_code(players_df, vice_code)}")
            else:
                st.info("Squad snapshot not available for this entry.")

            # ----- Raw JSON viewer (for debugging / support)
            with st.expander("Raw JSON (logged entry)", expanded=False):
                st.write(raw_entry)
