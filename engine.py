"""
Engine: parsing, metric derivation, guardrails, pacing and n-gram mining.
Kept separate from app.py so the logic is unit-testable without Streamlit.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(show_spinner=False)
except Exception:  # allows headless testing
    def _cache(fn):
        return fn


# ----------------------------------------------------------------------------
# 1. COLUMN CANONICALISATION
# Google Ads exports rename columns constantly between UI versions, report
# types and locales. Everything downstream depends on one canonical schema.
# ----------------------------------------------------------------------------

CANON = {
    "campaign": ["campaign", "campaign name"],
    "campaign_state": ["campaign state", "campaign status"],
    "campaign_type": ["campaign type", "advertising channel type", "campaign subtype"],
    "ad_group": ["ad group", "ad group name", "adgroup"],
    "ad_group_state": ["ad group state", "ad group status"],
    "keyword": ["keyword", "search keyword", "search keyword: keyword text", "keyword text", "criteria"],
    "match_type": ["match type", "keyword match type", "search keyword match type", "search term match type"],
    "search_term": ["search term", "search terms", "search term text"],
    "added_excluded": ["added/excluded", "added or excluded", "added / excluded"],
    "ad_label": ["ad", "ad name", "expanded text ad", "responsive search ad", "ad id", "ad type", "headline 1", "description line 1"],
    "final_url": ["final url", "final urls", "landing page", "ad final url"],
    "budget": ["budget", "daily budget", "campaign budget", "budget amount", "avg. daily budget", "average daily budget"],
    "cost": ["cost", "spend", "amount spent"],
    "clicks": ["clicks"],
    "impressions": ["impr.", "impr", "impressions"],
    "conversions": ["conversions", "conv.", "all conv.", "conversions (all)", "all conversions"],
    "conv_value": ["conv. value", "conversion value", "all conv. value", "total conv. value",
                   "conv. value (all)", "all conversion value", "revenue"],
    "date": ["day", "date"],
    "week": ["week"],
    "month": ["month"],
    "currency": ["currency code", "currency"],
    "search_is": ["search impr. share", "search impression share", "impr. share", "impression share"],
    "lost_is_budget": ["search lost is (budget)", "search lost impr. share (budget)",
                       "search lost top is (budget)", "lost is (budget)"],
    "lost_is_rank": ["search lost is (rank)", "search lost impr. share (rank)", "lost is (rank)"],
    "quality_score": ["quality score", "qual. score"],
    "bid_strategy": ["bid strategy type", "bid strategy", "bidding strategy type"],
}

# Reverse lookup built once
_LOOKUP = {}
for canon_name, variants in CANON.items():
    for v in variants:
        _LOOKUP[v] = canon_name

# Column tokens used to locate the real header row inside a messy export
HEADER_TOKENS = {"campaign", "ad group", "keyword", "search term", "clicks", "cost",
                 "impr.", "impressions", "conversions", "day", "budget", "ad"}

MONEY_RE = re.compile(r"[^\d\.\-]")


def _norm(col: str) -> str:
    c = str(col).replace("\ufeff", "").strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to canonical names; keep the rest untouched."""
    mapping = {}
    used = set()
    for col in df.columns:
        n = _norm(col)
        target = _LOOKUP.get(n)
        if target is None:
            # tolerate suffixes such as "Cost (USD)" or "Conversions (by conv. time)"
            base = re.sub(r"\s*\(.*\)\s*$", "", n).strip()
            target = _LOOKUP.get(base)
        if target and target not in used:
            mapping[col] = target
            used.add(target)
    return df.rename(columns=mapping)


def _to_number(series: pd.Series) -> pd.Series:
    """Clean Google's numeric strings: currency symbols, commas, %, --, < 10%, > 90%."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    is_pct = s.str.contains("%", na=False)

    # Impression-share sentinels
    s = s.str.replace("< 10%", "5%", regex=False).str.replace("<10%", "5%", regex=False)
    s = s.str.replace("> 90%", "95%", regex=False).str.replace(">90%", "95%", regex=False)

    s = s.replace({"--": np.nan, "-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    cleaned = s.str.replace(MONEY_RE, "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    out = out.where(~is_pct, out / 100.0)
    return out


NUMERIC_COLS = ["cost", "clicks", "impressions", "conversions", "conv_value", "budget",
                "search_is", "lost_is_budget", "lost_is_rank", "quality_score"]


def _read_raw(data: bytes) -> pd.DataFrame:
    """Handle UTF-8/UTF-16, comma/tab delimiters, and the 2-3 junk rows Google prepends."""
    last_err = None
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = data.decode(enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        if "\x00" in text[:2000]:
            continue

        lines = text.splitlines()
        if not lines:
            continue

        sample = "\n".join(lines[:40])
        sep = "\t" if sample.count("\t") > sample.count(",") else ","

        header_idx = 0
        for i, line in enumerate(lines[:15]):
            cells = {_norm(c) for c in line.split(sep)}
            if len(cells & HEADER_TOKENS) >= 2:
                header_idx = i
                break

        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=header_idx,
                             thousands=",", dtype=str, engine="python",
                             on_bad_lines="skip")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

        if df.shape[1] >= 2:
            return df

    raise ValueError(f"Could not parse file. Last error: {last_err}")


@_cache
def load_report(data: bytes, name: str) -> pd.DataFrame:
    df = _read_raw(data)
    df = _canonicalise(df)

    # Drop Google's "Total: ..." footer rows
    for key in ("campaign", "ad_group", "keyword", "search_term", "ad_label"):
        if key in df.columns:
            mask = df[key].astype(str).str.strip().str.lower().str.startswith("total")
            df = df[~mask]
            break

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = _to_number(df[c])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ("cost", "clicks", "impressions", "conversions", "conv_value"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    df = df.dropna(how="all")
    df.attrs["source"] = name
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. METRIC DERIVATION
# ----------------------------------------------------------------------------

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cpc"] = np.where(d["clicks"] > 0, d["cost"] / d["clicks"], np.nan)
    d["ctr"] = np.where(d["impressions"] > 0, d["clicks"] / d["impressions"], np.nan)
    d["cvr"] = np.where(d["clicks"] > 0, d["conversions"] / d["clicks"], np.nan)
    d["cpa"] = np.where(d["conversions"] > 0, d["cost"] / d["conversions"], np.nan)
    d["roas"] = np.where(d["cost"] > 0, d["conv_value"] / d["cost"], np.nan)
    d["aov"] = np.where(d["conversions"] > 0, d["conv_value"] / d["conversions"], np.nan)
    return d


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return pd.DataFrame()
    agg = {"cost": "sum", "clicks": "sum", "impressions": "sum",
           "conversions": "sum", "conv_value": "sum"}
    for extra, how in (("budget", "max"), ("search_is", "mean"),
                       ("lost_is_budget", "mean"), ("lost_is_rank", "mean"),
                       ("quality_score", "mean")):
        if extra in df.columns:
            agg[extra] = how
    out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
    return add_metrics(out)


# ----------------------------------------------------------------------------
# 3. THE GUARDRAIL — the reason this beats sorting a spreadsheet
# With hundreds of entities, most rows are too thin to judge. Acting on them
# is how accounts get destroyed. Every verdict is gated on expected conversions.
# ----------------------------------------------------------------------------

def verdict(df: pd.DataFrame, account_cvr: float, target_roas: float,
            target_cpa: float, min_expected_conv: float, use_roas: bool) -> pd.DataFrame:
    d = df.copy()
    d["expected_conv"] = d["clicks"] * max(account_cvr, 1e-9)
    d["judgeable"] = d["expected_conv"] >= min_expected_conv

    def _row(r):
        if not r["judgeable"] and r["conversions"] == 0:
            return "Not enough data", 0.0
        if r["conversions"] == 0:
            return "Zero conversions — cut or fix", float(r["cost"])
        if use_roas:
            if r["roas"] >= target_roas * 1.25:
                return "Above target — scale", 0.0
            if r["roas"] >= target_roas:
                return "On target — hold", 0.0
            # dollars misallocated = spend above what target ROAS would justify
            justified = r["conv_value"] / target_roas if target_roas > 0 else r["cost"]
            return "Below target ROAS", float(max(r["cost"] - justified, 0.0))
        else:
            if r["cpa"] <= target_cpa * 0.75:
                return "Below target CPA — scale", 0.0
            if r["cpa"] <= target_cpa:
                return "On target — hold", 0.0
            justified = r["conversions"] * target_cpa
            return "Above target CPA", float(max(r["cost"] - justified, 0.0))

    res = d.apply(_row, axis=1, result_type="expand")
    d["verdict"] = res[0]
    d["dollars_at_stake"] = res[1]
    return d.sort_values("dollars_at_stake", ascending=False)


VERDICT_COLOR = {
    "Zero conversions — cut or fix": "#b42318",
    "Below target ROAS": "#b54708",
    "Above target CPA": "#b54708",
    "On target — hold": "#475467",
    "Above target — scale": "#067647",
    "Below target CPA — scale": "#067647",
    "Not enough data": "#98a2b3",
}


# ----------------------------------------------------------------------------
# 4. PACING
# Google's real ceiling is daily_budget x 30.4 per month, not daily_budget x days.
# ----------------------------------------------------------------------------

MONTHLY_MULTIPLIER = 30.4


def pacing_table(camp: pd.DataFrame, days_in_data: int, days_in_month: int) -> pd.DataFrame:
    d = camp.copy()
    if "budget" not in d.columns:
        d["budget"] = np.nan
    d["avg_daily_spend"] = d["cost"] / max(days_in_data, 1)
    d["pace_vs_budget"] = np.where(d["budget"] > 0, d["avg_daily_spend"] / d["budget"], np.nan)
    d["projected_month_spend"] = d["avg_daily_spend"] * days_in_month
    d["monthly_ceiling"] = d["budget"] * MONTHLY_MULTIPLIER

    def _status(r):
        p = r["pace_vs_budget"]
        lost = r.get("lost_is_budget", np.nan)
        if pd.isna(p):
            return "No budget data"
        if p >= 0.90 or (pd.notna(lost) and lost >= 0.10):
            return "Budget capped"
        if p < 0.50:
            return "Severely underspending"
        if p < 0.75:
            return "Underspending"
        return "On pace"

    d["pacing_status"] = d.apply(_status, axis=1)
    return d


# ----------------------------------------------------------------------------
# 5. N-GRAM ENGINE — the only tractable way to mine search terms at scale
# ----------------------------------------------------------------------------

STOPWORDS = {"a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or", "by", "at"}


def ngrams(df: pd.DataFrame, text_col: str, n: int = 1, min_cost: float = 0.0) -> pd.DataFrame:
    rows = []
    sub = df[[text_col, "cost", "clicks", "impressions", "conversions", "conv_value"]].dropna(subset=[text_col])
    for rec in sub.itertuples(index=False):
        terms = re.findall(r"[a-z0-9\.\-\+/]+", str(getattr(rec, text_col)).lower())
        terms = [t for t in terms if t not in STOPWORDS]
        if len(terms) < n:
            continue
        seen = set()
        for i in range(len(terms) - n + 1):
            g = " ".join(terms[i:i + n])
            if g in seen:
                continue
            seen.add(g)
            rows.append((g, rec.cost, rec.clicks, rec.impressions, rec.conversions, rec.conv_value))
    if not rows:
        return pd.DataFrame()
    g = pd.DataFrame(rows, columns=["ngram", "cost", "clicks", "impressions", "conversions", "conv_value"])
    out = g.groupby("ngram", as_index=False).agg(
        cost=("cost", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"),
        conversions=("conversions", "sum"), conv_value=("conv_value", "sum"),
        terms=("ngram", "size"))
    out = out[out["cost"] >= min_cost]
    return add_metrics(out).sort_values("cost", ascending=False)


