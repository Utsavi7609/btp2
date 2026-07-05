"""
_taskI_divergence_test.py
==========================
SINGLE-SCRIPT PIPELINE — run this file to get all final results.

Phase 1: Physiological extraction + LLM inference (generates I2_val, I2_aro)
Phase 2: Body-Mind Divergence Test  (Hyp-2 / Hyp-6 + 4 supplementary tests)

On startup the script asks for your Groq API key.
If a key hits its quota / expires mid-run, it pauses and asks for a new key,
then retries the SAME row — no data is lost.
Progress is written row-by-row so a crash can always be resumed.

Run:  py -3 _taskI_divergence_test.py
Deps: pip install groq pandas numpy scipy
"""

import os, sys, json, time, glob, math, re
import numpy as np
import pandas as pd
from groq import Groq
from scipy.stats import fisher_exact, mannwhitneyu

# ── PATHS  (edit only if you moved files) ──────────────────────────────────

RESPONSES_CSV = r"D:\BTP\btp2main\clip_responses-export-2026-04-12_14-51-50.csv"
EMPATICA_DIR  = r"D:\BTP\btp2main\2lf92x651f5z6t6y"
EXPRESSED_CSV = r"D:\BTP\btp2\salmamaterials\final_expressed_valence_AUDIO.csv"
I2_CSV        = r"D:\BTP\btp2\salmamaterials\_validation_with_I2.csv"
RESULTS_CSV   = r"D:\BTP\btp2\salmamaterials\_taskI_divergence_results.csv"

# ── CONSTANTS ──────────────────────────────────────────────────────────────

LLM_MODEL    = "llama-3.3-70b-versatile"
TEMPERATURE  = 0.0
MAX_TOKENS   = 64
TAU          = 0.5       # equality threshold for continuous P/E/I1/I2 comparisons
WINDOW_PAD   = 60_000    # ms — extend clip window left by 1 minute (1-min Empatica buckets)

VALID_BLOCKS = {"Maintenance_Block", "Drift_Block",
                "Hyp9_Maintenance_Block", "Hyp9_Transition_Block"}

HARD = {("Q1","Q3"),("Q3","Q1"),("Q2","Q4"),("Q4","Q2")}
SOFT = {("Q1","Q2"),("Q2","Q1"),("Q2","Q3"),("Q3","Q2"),
        ("Q3","Q4"),("Q4","Q3"),("Q4","Q1"),("Q1","Q4")}

# ── GLOBAL (mutable) Groq client — replaced on key hot-swap ───────────────

GROQ_API_KEY = ""
client: Groq = None   # type: ignore


def init_client(key: str):
    global GROQ_API_KEY, client
    GROQ_API_KEY = key
    client = Groq(api_key=key)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — PHYSIOLOGICAL EXTRACTION + LLM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

# ── Empatica loader ────────────────────────────────────────────────────────

def load_signal(suffix: str) -> pd.DataFrame:
    """
    Recursively find all *_{suffix}.csv files, concatenate, drop bad rows,
    return DataFrame with columns [timestamp_unix (ms int), value (float)].
    """
    files = sorted(glob.glob(
        os.path.join(EMPATICA_DIR, "**", f"*_{suffix}.csv"), recursive=True))
    if not files:
        print(f"  [WARN] No *_{suffix}.csv files found under {EMPATICA_DIR}")
        return pd.DataFrame(columns=["timestamp_unix", "value"])

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["missing_value_reason"].isna()].copy()
        meta = {"timestamp_unix","timestamp_iso","participant_full_id","missing_value_reason"}
        val_col = next((c for c in df.columns if c not in meta), None)
        if val_col is None:
            continue
        df = df[["timestamp_unix", val_col]].rename(columns={val_col: "value"})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["timestamp_unix", "value"])

    out = pd.concat(dfs, ignore_index=True)
    out["timestamp_unix"] = out["timestamp_unix"].astype(np.int64)
    out["value"]          = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).sort_values("timestamp_unix").reset_index(drop=True)
    print(f"  {suffix:15s}: {len(out):4d} valid rows  "
          f"({pd.to_datetime(out['timestamp_unix'].iloc[0],  unit='ms', utc=True).strftime('%m-%d %H:%M')} "
          f"-> "
          f"{pd.to_datetime(out['timestamp_unix'].iloc[-1], unit='ms', utc=True).strftime('%m-%d %H:%M')} UTC)")
    return out


def window_stats(sig: pd.DataFrame, t0: int, t1: int) -> tuple:
    """Slice [t0 - WINDOW_PAD, t1], return (mean, max) or (nan, nan)."""
    sub = sig[(sig["timestamp_unix"] >= t0 - WINDOW_PAD) &
              (sig["timestamp_unix"] <= t1)]["value"]
    if len(sub) == 0:
        return float("nan"), float("nan")
    return float(sub.mean()), float(sub.max())


def extract_features(t0: int, t1: int, hr, prv, eda) -> dict:
    hr_mean,  _       = window_stats(hr,  t0, t1)
    hrv_rmssd, _      = window_stats(prv, t0, t1)
    eda_mean, eda_max = window_stats(eda, t0, t1)
    def r(x, n=2): return round(x, n) if not math.isnan(x) else float("nan")
    return {"hr_mean": r(hr_mean), "hrv_rmssd": r(hrv_rmssd),
            "eda_mean": r(eda_mean, 4), "eda_max": r(eda_max, 4)}


# ── Prompt builder ─────────────────────────────────────────────────────────

def fmt(v, unit):
    return f"{v:.2f} {unit}" if not math.isnan(v) else "unavailable"


def build_prompt(row: pd.Series, feats: dict) -> str:
    clip   = str(row.get("clip_title","?")).replace(".mp4","")
    dur    = (float(row["timestamp_clip_end"]) - float(row["timestamp_clip_start"])) / 1000
    pv, pa   = row.get("p_val","?"),  row.get("p_aro","?")
    i1v, i1a = row.get("i1_val","?"), row.get("i1_aro","?")
    return (
        "### Instruction\n"
        "You are an intelligent healthcare agent specialising in affective computing "
        "and wearable physiological signal analysis.\n\n"
        "### Health Knowledge\n"
        "Valence (emotional positivity): 1=very negative, 5=very positive.\n"
        "Arousal (activation level):     1=very calm, 5=very excited.\n"
        "Anchors: HR>90BPM + RMSSD<20ms => High Arousal, likely Negative Valence. "
        "HR65-80BPM + RMSSD>50ms => Low Arousal, Positive Valence. "
        "EDA>3uS => sympathetic activation (high arousal). EDA<0.5uS => parasympathetic (low arousal).\n\n"
        f"### Activity\nParticipant watched \"{clip}\" ({dur:.0f}s).\n\n"
        "### Self-Reports [context only -- do NOT copy as answer]\n"
        f"  Perceived (P):  Valence={pv}, Arousal={pa}\n"
        f"  Induced  (I1):  Valence={i1v}, Arousal={i1a}\n\n"
        "### Physiological Readings (Empatica EmbracePlus, 1-min aggregated)\n"
        f"  Mean Heart Rate (hr_mean)  : {fmt(feats['hr_mean'],   'BPM')}\n"
        f"  HRV RMSSD      (hrv_rmssd) : {fmt(feats['hrv_rmssd'], 'ms')}\n"
        f"  Mean EDA       (eda_mean)  : {fmt(feats['eda_mean'],  'uS')}\n"
        f"  Peak EDA       (eda_max)   : {fmt(feats['eda_max'],   'uS')}\n\n"
        "### Question\n"
        "Based solely on the physiological signals, predict the PHYSIOLOGICALLY INDUCED "
        "emotion (I2) experienced during this clip. Use the 1-5 integer scale.\n\n"
        "### Response\n"
        'Return ONLY this JSON, no extra text: {"I2_val": <int 1-5>, "I2_aro": <int 1-5>}'
    )


# ── LLM call with key-exhaustion hot-swap ──────────────────────────────────

def call_llm(prompt: str, row_tag: str) -> dict | None:
    """
    Call Groq with retry logic:
      429 / rate limit  -> exponential back-off (up to 5 attempts)
      401 / auth / quota -> PAUSE and ask user for a new API key, then retry same row
      other errors      -> retry up to 5 times
    Returns {"I2_val": int, "I2_aro": int} or None on total failure.
    """
    retries = 0

    while retries < 5:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},   # Groq guarantees valid JSON
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw       = resp.choices[0].message.content.strip()
            raw_clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
            parsed    = json.loads(raw_clean)

            i2v = int(round(float(parsed.get("I2_val", parsed.get("valence",  3)))))
            i2a = int(round(float(parsed.get("I2_aro", parsed.get("arousal",  3)))))
            return {"I2_val": max(1, min(5, i2v)), "I2_aro": max(1, min(5, i2a))}

        except Exception as e:
            err = str(e)

            # ── Key exhaustion: pause and hot-swap ───────────────────────
            if any(tok in err.lower() for tok in
                   ("401", "auth", "api_key", "quota", "invalid_api_key")):
                print(f"\n\n  [KEY EXHAUSTED / INVALID] {row_tag}")
                print(f"  Error: {err}")
                new_key = input(
                    "\n  >>> Current key failed. Paste a NEW Groq API key and press Enter: "
                ).strip()
                if new_key:
                    init_client(new_key)
                    print("  >>> Key updated. Retrying same row ...\n")
                else:
                    print("  >>> No key provided. Skipping row.")
                    return None
                continue   # retry same row, does NOT increment retries

            # ── Rate limit (429): exponential back-off ─────────────────
            elif "429" in err or "rate" in err.lower():
                wait = 45 * (retries + 1)
                print(f"\n  [Rate limit] Waiting {wait}s ... (attempt {retries+1}/5)")
                time.sleep(wait)

            # ── Bad JSON ───────────────────────────────────────────────
            elif "json" in err.lower():
                print(f"\n  [JSON error] attempt {retries+1}/5  raw={err[:120]}")
                time.sleep(10)

            # ── Anything else ──────────────────────────────────────────
            else:
                print(f"\n  [Error] attempt {retries+1}/5  {err[:120]}")
                time.sleep(15)

            retries += 1

    print(f"  [SKIP] {row_tag} — failed after 5 retries.")
    return None


# ── Expressed emotion lookup ───────────────────────────────────────────────

def load_expressed(path: str) -> dict:
    df = pd.read_csv(path)
    return {str(r["clip_id"]): (float(r["expressed_valence"]), float(r["expressed_arousal"]))
            for _, r in df.iterrows()}


def equad(clip_title: str, e_dict: dict) -> str:
    cid = str(clip_title).replace(".mp4","").strip()
    e = e_dict.get(cid)
    if e is None: return "Q0"
    v, a = e
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"


# ── Phase 1 entry point ────────────────────────────────────────────────────

def phase1_infer_i2():
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  PHASE 1 — Physiological Extraction + LLM Inference (I2)")
    print(SEP)

    # ── Load Empatica continuous stream ───────────────────────────────────
    print("\n  Loading Empatica data stream ...")
    hr_df  = load_signal("pulse-rate")
    prv_df = load_signal("prv")
    eda_df = load_signal("eda")

    phys_start = int(hr_df["timestamp_unix"].min()) if len(hr_df) else 0
    phys_end   = int(hr_df["timestamp_unix"].max()) if len(hr_df) else 0

    # ── Load and filter responses ─────────────────────────────────────────
    df = pd.read_csv(RESPONSES_CSV, sep=";")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df[df["block_type"].isin(VALID_BLOCKS)].copy()

    # ── Add ALL output/derived columns to df BEFORE any slicing ──────────
    # CRITICAL: df_phys must have the exact same column set as the CSV header.
    # If these columns are added after slicing, appended rows will be misaligned
    # (I2_aro value lands in I2_val column, I2_aro column stays NaN).
    e_dict = load_expressed(EXPRESSED_CSV)
    df["E_quadrant"] = df["clip_title"].apply(lambda t: equad(t, e_dict))
    if "I2_val" not in df.columns:
        df["I2_val"] = float("nan")
    if "I2_aro" not in df.columns:
        df["I2_aro"] = float("nan")
    if "phys_hr_mean" not in df.columns:
        df["phys_hr_mean"] = float("nan")
    if "phys_eda_mean" not in df.columns:
        df["phys_eda_mean"] = float("nan")

    # ── NOW mark physio window and slice ──────────────────────────────────
    df["in_phys"] = (
        (df["timestamp_clip_end"]   >= phys_start - WINDOW_PAD) &
        (df["timestamp_clip_start"] <= phys_end)
    )
    df_phys   = df[df["in_phys"]].copy()   # NOW has E_quadrant, I2_val, I2_aro, phys_*
    df_nophys = df[~df["in_phys"]].copy()
    print(f"\n  Rows inside  Empatica window : {len(df_phys)}  (will get I2 via LLM)")
    print(f"  Rows outside Empatica window : {len(df_nophys)}  (I2 will be NaN)")

    # ── Resume logic: check how many rows already done ────────────────────
    if os.path.exists(I2_CSV):
        df_done  = pd.read_csv(I2_CSV)
        done_ids = set(df_done["id"].tolist()) if "id" in df_done.columns else set()
        print(f"\n  Resume: {len(done_ids)} rows already in {I2_CSV}")
    else:
        df_done  = None
        done_ids = set()
        # Write header now — df already has ALL columns, so the order is correct
        pd.DataFrame(columns=list(df.columns)).to_csv(I2_CSV, index=False)
        print(f"\n  Starting fresh — output file created: {I2_CSV}")

    # ── Inference loop ────────────────────────────────────────────────────
    rows_to_do = df_phys[~df_phys["id"].isin(done_ids)] if "id" in df_phys.columns else df_phys
    total = len(rows_to_do)
    done  = 0

    if total == 0:
        print("\n  All rows already processed. Skipping LLM calls.")
    else:
        print(f"\n  Running LLM on {total} rows ...\n")

    for _, row in rows_to_do.iterrows():
        done += 1
        tag  = f"[{done:3d}/{total}] user={int(row['user_id']):2d}  {str(row['clip_title']):<35s}"

        t0 = int(row["timestamp_clip_start"])
        t1 = int(row["timestamp_clip_end"])

        feats   = extract_features(t0, t1, hr_df, prv_df, eda_df)
        all_nan = all(math.isnan(v) for v in feats.values())

        # row already has E_quadrant, I2_val=NaN, I2_aro=NaN, phys_*=NaN
        row_out = row.copy()

        if all_nan:
            print(f"  {tag} -> no data in window, I2=NaN")
            pd.DataFrame([row_out]).to_csv(I2_CSV, mode="a", header=False, index=False)
            continue

        print(f"  {tag}")
        print(f"    hr={feats['hr_mean']} bpm | rmssd={feats['hrv_rmssd']} ms "
              f"| eda={feats['eda_mean']}/{feats['eda_max']} uS")

        # Save physio features so Phase 2 can run direct physiological tests
        row_out["phys_hr_mean"]  = feats["hr_mean"]
        row_out["phys_eda_mean"] = feats["eda_mean"]

        result = call_llm(build_prompt(row, feats), tag)
        if result:
            row_out["I2_val"] = result["I2_val"]
            row_out["I2_aro"] = result["I2_aro"]
            print(f"    -> I2_val={result['I2_val']}, I2_aro={result['I2_aro']}")

        # Append row immediately (survive any crash / resume on restart)
        pd.DataFrame([row_out]).to_csv(I2_CSV, mode="a", header=False, index=False)
        time.sleep(1.2)    # polite rate-limit buffer

    # ── Append the no-physio rows so the full dataset is in one file ──────
    if len(df_nophys) > 0:
        already = set(pd.read_csv(I2_CSV)["id"].tolist()) if "id" in df_nophys.columns else set()
        missing = df_nophys[~df_nophys["id"].isin(already)] if "id" in df_nophys.columns else df_nophys
        if len(missing):
            missing.to_csv(I2_CSV, mode="a", header=False, index=False)

    final = pd.read_csv(I2_CSV)
    i2_ok = final["I2_val"].notna().sum()
    print(f"\n  Phase 1 complete. Total rows: {len(final)}")
    print(f"  I2 populated: {i2_ok}  |  NaN: {len(final)-i2_ok}")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — BODY-MIND DIVERGENCE TEST  (Hyp-2 / Hyp-6 + supplementary)
# ═══════════════════════════════════════════════════════════════════════════

def get_quad(v, a):
    try:
        v, a = float(v), float(a)
    except (TypeError, ValueError):
        return "Q0"
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"


def eq(va, aa, vb, ab, tau=TAU):
    try:
        return abs(float(va)-float(vb)) <= tau and abs(float(aa)-float(ab)) <= tau
    except (TypeError, ValueError):
        return False


def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id","created_at"]).copy()
    rows = []
    for uid, udf in df.groupby("user_id", sort=False):
        udf  = udf.sort_values("clip_position").reset_index()
        sess, prev = 0, None
        for _, r in udf.iterrows():
            pos = r["clip_position"]
            if prev is None or pos <= prev:
                sess += 1
            rows.append((r["index"], sess))
            prev = pos
    df["true_session"] = df.index.map(dict(rows))
    return df


def build_pairs(df: pd.DataFrame, e_dict: dict) -> pd.DataFrame:
    records = []
    df = assign_sessions(df)

    for (uid, sess, block), grp in df.groupby(
            ["user_id","true_session","block_type"], sort=False):
        if block not in VALID_BLOCKS:
            continue
        c1r = grp[grp["expected_role"] == "Clip1_Stimulus"]
        c2r = grp[grp["expected_role"] == "Clip2_Target"]
        if c1r.empty or c2r.empty:
            continue

        c1, c2 = c1r.iloc[0], c2r.iloc[0]

        e1 = e_dict.get(str(c1["clip_title"]).replace(".mp4",""))
        e2 = e_dict.get(str(c2["clip_title"]).replace(".mp4",""))
        if e1 is None or e2 is None:
            continue

        q1, q2 = get_quad(*e1), get_quad(*e2)
        pair   = (q1, q2)

        if   pair in HARD: t_type = "Hard"
        elif pair in SOFT: t_type = "Soft"
        else:              continue   # same quad or Q0

        records.append({
            "user_id":         uid,
            "true_session":    sess,
            "block_type":      block,
            "clip1_title":     c1["clip_title"],
            "clip1_E_quad":    q1,
            "clip2_title":     c2["clip_title"],
            "clip2_E_quad":    q2,
            "E_val":           e2[0],   "E_aro":   e2[1],
            "P_val":           c2["p_val"],   "P_aro":   c2["p_aro"],
            "I1_val":          c2["i1_val"],  "I1_aro":  c2["i1_aro"],
            "I2_val":          c2.get("I2_val",       float("nan")),
            "I2_aro":          c2.get("I2_aro",       float("nan")),
            "phys_hr_mean":    c2.get("phys_hr_mean",  float("nan")),
            "phys_eda_mean":   c2.get("phys_eda_mean", float("nan")),
            "transition_type": t_type,
        })

    return pd.DataFrame(records)


# ── Classification functions ───────────────────────────────────────────────

def is_b3nb6(row: pd.Series) -> bool:
    """
    TEST 1 — Continuous: P==E  AND  P!=I1  AND  P==I2  (all within tau on both dims).
    The original Hyp-2/Hyp-6 formulation.
    """
    try:
        pv,pa   = float(row["P_val"]),  float(row["P_aro"])
        ev,ea   = float(row["E_val"]),  float(row["E_aro"])
        i1v,i1a = float(row["I1_val"]), float(row["I1_aro"])
        i2v,i2a = float(row["I2_val"]), float(row["I2_aro"])
    except (TypeError, ValueError):
        return False
    if any(x != x for x in [pv,pa,ev,ea,i1v,i1a,i2v,i2a]):  # NaN guard
        return False
    return eq(pv,pa,ev,ea) and (not eq(pv,pa,i1v,i1a)) and eq(pv,pa,i2v,i2a)


def is_b3nb6_quad(row: pd.Series) -> bool:
    """
    TEST 2 — Quadrant-level: P_quad==E_quad  AND  P_quad!=I1_quad  AND  P_quad==I2_quad.
    Coarser (4-cell) version — more lenient, better statistical power.
    """
    p_quad  = get_quad(row["P_val"],  row["P_aro"])
    e_quad  = row["clip2_E_quad"]
    i1_quad = get_quad(row["I1_val"], row["I1_aro"])
    i2_quad = get_quad(row["I2_val"], row["I2_aro"])
    if "Q0" in (p_quad, e_quad, i1_quad, i2_quad):
        return False
    return (p_quad == e_quad) and (p_quad != i1_quad) and (p_quad == i2_quad)


def closer_i2(row: pd.Series) -> bool:
    """
    TEST 3 — Hyp-7 proxy: I2 is closer to E than I1 in Euclidean val-aro space.
    Hypothesis: body (I2) tracks expressed emotion better than survey (I1),
    especially for Hard (opposing-quadrant) transitions where conscious denial peaks.
    """
    try:
        ev, ea   = float(row["E_val"]),  float(row["E_aro"])
        i1v, i1a = float(row["I1_val"]), float(row["I1_aro"])
        i2v, i2a = float(row["I2_val"]), float(row["I2_aro"])
        if any(math.isnan(x) for x in [ev,ea,i1v,i1a,i2v,i2a]):
            return False
        d_i1 = math.sqrt((i1v-ev)**2 + (i1a-ea)**2)
        d_i2 = math.sqrt((i2v-ev)**2 + (i2a-ea)**2)
        return d_i2 < d_i1
    except (TypeError, ValueError):
        return False


# ── Helper: compact Fisher test block ─────────────────────────────────────

def fisher_report(label: str,
                  hard_yes: int, hard_n: int,
                  soft_yes: int, soft_n: int) -> tuple:
    hr = hard_yes / hard_n if hard_n else 0.0
    sr = soft_yes / soft_n if soft_n else 0.0
    table = [[hard_yes, hard_n - hard_yes], [soft_yes, soft_n - soft_yes]]
    odds, p = fisher_exact(table, alternative="greater")
    sig = "*** p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "n.s.")
    print(f"    Hard : {hard_yes}/{hard_n}  ({100*hr:.1f}%)")
    print(f"    Soft : {soft_yes}/{soft_n}  ({100*sr:.1f}%)")
    print(f"    Fisher's Exact (one-sided Hard>Soft): p={p:.4f}  OR={odds:.3f}  [{sig}]")
    return p, odds


# ── Phase 2 entry point ────────────────────────────────────────────────────

def phase2_divergence_test():
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  PHASE 2 — Body-Mind Divergence Tests  (Hyp-2 / Hyp-6)")
    print(SEP)

    df = pd.read_csv(I2_CSV)
    print(f"\n  Loaded {len(df)} rows from {I2_CSV}")

    # Diagnostic: both I2_val and I2_aro must be populated together
    i2v_ok = int(df["I2_val"].notna().sum())
    i2a_ok = int(df["I2_aro"].notna().sum())
    print(f"  I2 coverage — I2_val: {i2v_ok}  I2_aro: {i2a_ok}  "
          f"({'OK' if i2v_ok == i2a_ok else 'MISMATCH — column alignment error?'})")
    if i2v_ok == 0:
        print("\n  WARNING: no I2 values found. All tests will show zero counts.")

    e_dict = load_expressed(EXPRESSED_CSV)
    pairs  = build_pairs(df, e_dict)

    n_hard = int((pairs["transition_type"] == "Hard").sum())
    n_soft = int((pairs["transition_type"] == "Soft").sum())
    print(f"\n  Reconstructed {len(pairs)} Clip1->Clip2 pairs")
    print(f"  Hard transitions (Q1<->Q3, Q2<->Q4) : {n_hard}")
    print(f"  Soft transitions (adjacent)          : {n_soft}")

    if len(pairs) == 0:
        sys.exit("  ERROR: no pairs found. Check block_type / expected_role values.")

    # Compute ALL flag columns BEFORE slicing — slices are views and won't
    # see columns added to the parent DataFrame after the slice is taken.
    pairs["B3_n_B6"]      = pairs.apply(is_b3nb6,      axis=1)
    pairs["B3_n_B6_quad"] = pairs.apply(is_b3nb6_quad, axis=1)
    pairs["closer_I2"]    = pairs.apply(closer_i2,     axis=1)

    hard_df = pairs[pairs["transition_type"] == "Hard"].copy()
    soft_df = pairs[pairs["transition_type"] == "Soft"].copy()
    hn, sn  = len(hard_df), len(soft_df)

    # ── TEST 1: {B3 n B6} continuous ─────────────────────────────────────
    print(f"\n  {'-'*63}")
    print(f"  TEST 1 — {{B3 n B6}} continuous  "
          f"(P==E AND P!=I1 AND P==I2, tau={TAU})")
    fisher_report("B3nB6-continuous",
                  int(hard_df["B3_n_B6"].sum()), hn,
                  int(soft_df["B3_n_B6"].sum()), sn)

    # ── TEST 2: {B3 n B6} quadrant-level ─────────────────────────────────
    print(f"\n  {'-'*63}")
    print(f"  TEST 2 — {{B3 n B6}} quadrant    "
          f"(P_quad==E_quad AND P_quad!=I1_quad AND P_quad==I2_quad)")
    fisher_report("B3nB6-quadrant",
                  int(hard_df["B3_n_B6_quad"].sum()), hn,
                  int(soft_df["B3_n_B6_quad"].sum()), sn)

    # ── TEST 3: Hyp-7 proxy — I2 closer to E than I1 ─────────────────────
    pairs_i2 = pairs[pairs["I2_val"].notna() & pairs["I2_aro"].notna()]
    h3_hard  = pairs_i2[pairs_i2["transition_type"] == "Hard"]
    h3_soft  = pairs_i2[pairs_i2["transition_type"] == "Soft"]
    print(f"\n  {'-'*63}")
    print(f"  TEST 3 — Hyp-7 proxy: I2 closer to E than I1  "
          f"(body tracks expressed emotion better than survey)")
    if len(h3_hard) > 0 and len(h3_soft) > 0:
        fisher_report("Hyp7-I2-closer",
                      int(h3_hard["closer_I2"].sum()), len(h3_hard),
                      int(h3_soft["closer_I2"].sum()), len(h3_soft))
    else:
        print(f"    Insufficient pairs with I2  (hard={len(h3_hard)}, soft={len(h3_soft)})")

    # ── TEST 4: Direct physiological — Mean HR ────────────────────────────
    print(f"\n  {'-'*63}")
    print(f"  TEST 4 — Direct physiological: Mean HR  "
          f"(Hard vs Soft, Mann-Whitney two-sided)")
    hard_hr = hard_df["phys_hr_mean"].dropna()
    soft_hr = soft_df["phys_hr_mean"].dropna()
    if len(hard_hr) >= 3 and len(soft_hr) >= 3:
        stat, p_hr = mannwhitneyu(hard_hr, soft_hr, alternative="two-sided")
        sig = "*** p<0.05" if p_hr < 0.05 else ("~ p<0.10" if p_hr < 0.10 else "n.s.")
        print(f"    Hard HR : n={len(hard_hr):3d}  mean={hard_hr.mean():.1f} BPM  "
              f"median={hard_hr.median():.1f}")
        print(f"    Soft HR : n={len(soft_hr):3d}  mean={soft_hr.mean():.1f} BPM  "
              f"median={soft_hr.median():.1f}")
        print(f"    Mann-Whitney p={p_hr:.4f}  [{sig}]")
    else:
        print(f"    Insufficient physio data  "
              f"(hard HR n={len(hard_hr)}, soft HR n={len(soft_hr)})")

    # ── TEST 5: Direct physiological — Mean EDA ───────────────────────────
    print(f"\n  {'-'*63}")
    print(f"  TEST 5 — Direct physiological: Mean EDA  "
          f"(Hard vs Soft, Mann-Whitney two-sided)")
    hard_eda = hard_df["phys_eda_mean"].dropna()
    soft_eda = soft_df["phys_eda_mean"].dropna()
    if len(hard_eda) >= 3 and len(soft_eda) >= 3:
        stat, p_eda = mannwhitneyu(hard_eda, soft_eda, alternative="two-sided")
        sig = "*** p<0.05" if p_eda < 0.05 else ("~ p<0.10" if p_eda < 0.10 else "n.s.")
        print(f"    Hard EDA: n={len(hard_eda):3d}  mean={hard_eda.mean():.4f} uS  "
              f"median={hard_eda.median():.4f}")
        print(f"    Soft EDA: n={len(soft_eda):3d}  mean={soft_eda.mean():.4f} uS  "
              f"median={soft_eda.median():.4f}")
        print(f"    Mann-Whitney p={p_eda:.4f}  [{sig}]")
    else:
        print(f"    Insufficient physio data  "
              f"(hard EDA n={len(hard_eda)}, soft EDA n={len(soft_eda)})")

    pairs.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Pair results -> {RESULTS_CSV}")
    print(f"  {SEP}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  BTP-2 Validation Pipeline  —  Hyp-2 / Hyp-6")
    print("=" * 65)

    # Ask for API key upfront
    key_input = input(
        "\n  Enter your Groq API key (starts with gsk_): "
    ).strip()
    if not key_input:
        sys.exit("  No API key provided. Exiting.")
    init_client(key_input)
    print(f"  Key set: {key_input[:8]}...{key_input[-4:]}\n")

    # Phase 1: extract physio features + infer I2 via LLM
    phase1_infer_i2()

    # Phase 2: all divergence / physiological tests
    phase2_divergence_test()
