"""
Google Ads Triage Console
=========================
Built for accounts too large to review campaign-by-campaign.

This is not a reporting dashboard. It is a decision engine: it reads standard
Google Ads CSV exports, applies statistical guardrails, and returns a ranked
action list with dollars-at-stake attached to every row.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Google Ads Triage Console", layout="wide", page_icon="⚡")

from engine import (  # noqa: E402
    CANON, MONTHLY_MULTIPLIER, aggregate, load_report, ngrams, pacing_table, verdict,
)


class SimpleUpload:
    """Gives a file on disk the same interface as a Streamlit upload."""

    def __init__(self, path: Path):
        self._path = path
        self.name = path.name

    def getvalue(self) -> bytes:
        return self._path.read_bytes()

# ----------------------------------------------------------------------------
# 6. UI
# ----------------------------------------------------------------------------

def money(x, sym="$"):
    if pd.isna(x):
        return "—"
    return f"{sym}{x:,.0f}"


def fmt_frame(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    d = df.copy()
    for c in ("cost", "conv_value", "cpa", "cpc", "aov", "budget", "avg_daily_spend",
              "projected_month_spend", "monthly_ceiling", "dollars_at_stake"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: money(v, sym))
    for c in ("ctr", "cvr", "search_is", "lost_is_budget", "lost_is_rank", "pace_vs_budget"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v*100:,.1f}%")
    for c in ("roas",):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}x")
    for c in ("conversions", "expected_conv"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.1f}")
    for c in ("clicks", "impressions"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
    return d


def dl_button(df: pd.DataFrame, label: str, fname: str, key: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv", key=key)


st.title("⚡ Google Ads Triage Console")
st.caption("Ranked action list, not a report. Every row carries the dollars at stake.")

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("1 · Load exports")
    st.caption("Drop in your Google Ads CSVs. Filenames are auto-classified; "
               "override below if anything lands in the wrong bucket.")
    source = st.radio("Source", ["Upload files", "Read from folder"], index=0, horizontal=True)
    uploads = []
    if source == "Upload files":
        uploads = st.file_uploader("Google Ads CSV exports", type=["csv", "tsv", "txt"],
                                   accept_multiple_files=True) or []
    else:
        folder = st.text_input("Folder path", value="sample_data",
                               help="Point this at the folder you drop the weekly exports into. "
                                    "Every .csv inside is read.")
        p = Path(folder).expanduser()
        if p.is_dir():
            found = sorted(list(p.glob("*.csv")) + list(p.glob("*.tsv")))
            st.caption(f"{len(found)} file(s) found")
            uploads = [SimpleUpload(f) for f in found]
        elif folder:
            st.error("Folder not found.")

    st.header("2 · Economics")
    sym = st.text_input("Currency symbol", value="$", max_chars=3)
    goal = st.radio("Primary goal", ["ROAS (ecommerce)", "CPA (lead gen)"], index=0)
    use_roas = goal.startswith("ROAS")
    target_roas = st.number_input("Target ROAS (x)", value=4.0, min_value=0.1, step=0.5,
                                  help="Break-even ROAS = 1 / gross margin. At 25% margin, break-even is 4.0x.")
    target_cpa = st.number_input(f"Target CPA ({sym})", value=95.0, min_value=1.0, step=5.0)
    gross_margin = st.slider("Gross margin %", 5, 90, 25,
                             help="Used to convert revenue into contribution profit.") / 100

    st.header("3 · Guardrails")
    min_expected_conv = st.slider("Minimum expected conversions before judging", 1.0, 10.0, 3.0, 0.5,
                                  help="Entities with fewer expected conversions than this are marked "
                                       "'Not enough data'. This is what stops you cutting 200 campaigns "
                                       "on statistical noise.")
    min_waste_cost = st.number_input(f"Ignore entities below this spend ({sym})", value=25.0, min_value=0.0, step=5.0)

    st.header("4 · Period")
    auto_days = st.checkbox("Detect date range from file", value=True)
    manual_days = st.number_input("Days in the export", value=30, min_value=1, max_value=400)
    days_in_month = st.number_input("Days in pacing month", value=30, min_value=28, max_value=31)
    monthly_cap = st.number_input(f"Account monthly budget cap ({sym}, 0 = none)", value=0.0, min_value=0.0, step=250.0)

if not uploads:
    st.info("Upload at least the **campaign-level** export to begin. "
            "Add ad group, keyword, search term and ad exports to unlock the deeper tabs.")
    with st.expander("Which columns each export needs"):
        st.markdown(
            """
| Export | Required columns | Unlocks |
|---|---|---|
| Campaign | Campaign, Budget, Cost, Clicks, Impr., Conversions, Conv. value | Pacing, account health, scale/cut |
| Ad group | Campaign, Ad group, Cost, Clicks, Conversions, Conv. value | Ad group triage |
| Keyword | Campaign, Ad group, Keyword, Match type, Cost, Clicks, Conversions, Conv. value | Match-type economics, keyword cuts |
| Search terms | Search term, Campaign, Cost, Clicks, Conversions, Conv. value, Added/Excluded | N-gram waste mining, negative lists |
| Ads | Campaign, Ad group, Ad, Cost, Clicks, Impr., Conversions, Conv. value | Creative triage |

Optional but high value on the campaign export: **Search lost IS (budget)**, **Search lost IS (rank)**,
**Search impr. share**, **Bid strategy type**, **Day**.
            """)
    st.stop()

# ---- Classify uploads --------------------------------------------------------
loaded: dict[str, pd.DataFrame] = {}
raw = {}
for up in uploads:
    try:
        raw[up.name] = load_report(up.getvalue(), up.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read **{up.name}** — {e}")


def classify(name: str, df: pd.DataFrame) -> str:
    n = name.lower()
    cols = set(df.columns)
    if "search_term" in cols or "search term" in n:
        return "search_terms"
    if "keyword" in cols or "keyword" in n:
        return "keywords"
    if "ad_label" in cols and "ad_group" in cols and ("ad" in n or "creative" in n):
        return "ads"
    if "ad_group" in cols:
        return "ad_groups"
    if "campaign" in cols:
        return "campaigns"
    return "unknown"


with st.sidebar:
    st.header("5 · File mapping")
    options = ["campaigns", "ad_groups", "keywords", "search_terms", "ads", "ignore"]
    for name, df in raw.items():
        guess = classify(name, df)
        idx = options.index(guess) if guess in options else len(options) - 1
        choice = st.selectbox(f"{name[:34]}", options, index=idx, key=f"map_{name}")
        if choice != "ignore":
            loaded[choice] = pd.concat([loaded[choice], df]) if choice in loaded else df

if "campaigns" not in loaded:
    st.warning("No campaign-level export detected. Map one in the sidebar — the account "
               "health and pacing views depend on it.")

# ---- Date range --------------------------------------------------------------
days_in_data = manual_days
detected = None
for df in loaded.values():
    if "date" in df.columns and df["date"].notna().any():
        span = (df["date"].max() - df["date"].min()).days + 1
        detected = (df["date"].min().date(), df["date"].max().date(), span)
        break
if auto_days and detected:
    days_in_data = detected[2]
    st.caption(f"Date range detected: **{detected[0]} → {detected[1]}** ({days_in_data} days)")
else:
    st.caption(f"Using **{days_in_data} days** for the export period (set in sidebar).")

# ---- Account totals ----------------------------------------------------------
base = loaded.get("campaigns")
if base is None:
    base = next(iter(loaded.values()))

camp = aggregate(base, ["campaign", "campaign_state", "campaign_type", "bid_strategy"])
if camp.empty:
    camp = aggregate(base, ["campaign"])

tot_cost = camp["cost"].sum()
tot_conv = camp["conversions"].sum()
tot_val = camp["conv_value"].sum()
tot_clicks = camp["clicks"].sum()
account_cvr = tot_conv / tot_clicks if tot_clicks else 0.0
account_roas = tot_val / tot_cost if tot_cost else np.nan
account_cpa = tot_cost / tot_conv if tot_conv else np.nan
gross_profit = tot_val * gross_margin - tot_cost

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Spend", money(tot_cost, sym))
c2.metric("Revenue (platform)", money(tot_val, sym))
c3.metric("ROAS", "—" if pd.isna(account_roas) else f"{account_roas:,.2f}x",
          delta=None if pd.isna(account_roas) else f"{account_roas - target_roas:+.2f} vs target")
c4.metric("CPA", money(account_cpa, sym))
c5.metric("Conversions", f"{tot_conv:,.0f}")
c6.metric("Contribution profit", money(gross_profit, sym),
          help=f"Revenue x {gross_margin:.0%} margin − ad spend. Negative means the account is buying revenue at a loss.")

# Concentration — reframes the "too many campaigns" problem
camp_sorted = camp.sort_values("cost", ascending=False)
n_camp = len(camp_sorted)
top10_share = camp_sorted.head(10)["cost"].sum() / tot_cost if tot_cost else 0
n_spending = int((camp_sorted["cost"] > 0).sum())
n_zero_conv = int(((camp_sorted["conversions"] == 0) & (camp_sorted["cost"] > min_waste_cost)).sum())
zero_conv_spend = camp_sorted.loc[(camp_sorted["conversions"] == 0) &
                                  (camp_sorted["cost"] > min_waste_cost), "cost"].sum()

st.markdown(
    f"**{n_camp:,} campaigns in file · {n_spending:,} actually spent · "
    f"top 10 hold {top10_share:.0%} of spend · "
    f"{n_zero_conv:,} campaigns burned {money(zero_conv_spend, sym)} with zero conversions.**"
)

tabs = st.tabs(["🎯 Action list", "⏱️ Pacing", "📊 Campaigns", "🧩 Ad groups",
                "🔑 Keywords", "🔎 Search terms", "🖼️ Ads", "🧪 Data check"])

# ============================================================== ACTION LIST ===
with tabs[0]:
    st.subheader("Do these first")
    st.caption("Sorted by dollars at stake — the spend that would be freed or the revenue "
               "unlocked if you acted. Everything below the guardrail is excluded.")

    frames = []
    level_map = [("campaigns", ["campaign"], "Campaign"),
                 ("ad_groups", ["campaign", "ad_group"], "Ad group"),
                 ("keywords", ["campaign", "ad_group", "keyword", "match_type"], "Keyword"),
                 ("search_terms", ["search_term", "campaign"], "Search term")]

    for key, group_keys, label in level_map:
        if key not in loaded:
            continue
        agg = aggregate(loaded[key], group_keys)
        if agg.empty:
            continue
        agg = agg[agg["cost"] >= min_waste_cost]
        if agg.empty:
            continue
        v = verdict(agg, account_cvr, target_roas, target_cpa, min_expected_conv, use_roas)
        v.insert(0, "level", label)
        v["entity"] = v[[k for k in group_keys if k in v.columns]].astype(str).agg(" › ".join, axis=1)
        frames.append(v)

    if frames:
        actions = pd.concat(frames, ignore_index=True)
        act_cols = ["level", "entity", "verdict", "dollars_at_stake", "cost", "conv_value",
                    "roas", "cpa", "conversions", "clicks", "expected_conv"]
        act_cols = [c for c in act_cols if c in actions.columns]

        colf1, colf2 = st.columns([2, 1])
        with colf1:
            pick = st.multiselect("Verdicts", sorted(actions["verdict"].unique()),
                                  default=[v for v in actions["verdict"].unique() if v != "Not enough data"])
        with colf2:
            levels = st.multiselect("Levels", sorted(actions["level"].unique()),
                                    default=sorted(actions["level"].unique()))

        view = actions[actions["verdict"].isin(pick) & actions["level"].isin(levels)]
        view = view.sort_values("dollars_at_stake", ascending=False)

        # The same dollar appears at campaign, ad group, keyword and search term level.
        # Summing across levels would inflate the number 3-4x, so report per level.
        bad = view["verdict"].str.contains("Zero|Below|Above target CPA")
        good = view["verdict"].str.contains("scale")
        per_level = pd.DataFrame({
            "Recoverable spend": view[bad].groupby("level")["dollars_at_stake"].sum(),
            "Spend in scale rows": view[good].groupby("level")["cost"].sum(),
            "Rows": view.groupby("level").size(),
        }).fillna(0)
        order = [l for l in ["Campaign", "Ad group", "Keyword", "Search term"] if l in per_level.index]
        per_level = per_level.loc[order]

        headline = per_level["Recoverable spend"].iloc[0] if len(per_level) else 0
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Recoverable spend ({order[0] if order else '—'} level)", money(headline, sym),
                  help="Spend not clearing your target at this level. Not all of it is savable — "
                       "it is the pool you are negotiating with.")
        m2.metric("Spend in 'scale' rows", money(per_level['Spend in scale rows'].iloc[0] if len(per_level) else 0, sym))
        m3.metric("Rows shown", f"{len(view):,}")
        st.dataframe(per_level.map(lambda v: f"{v:,.0f}"), width="stretch")
        st.caption("Each level re-cuts the same spend. Read one level at a time — never add them together.")

        st.dataframe(fmt_frame(view[act_cols].head(400), sym), width="stretch", hide_index=True)
        dl_button(view[act_cols], "⬇ Download full action list (CSV)", "gads_action_list.csv", "dl_actions")

        st.markdown("##### Where the money sits")
        pivot = (view.groupby(["level", "verdict"], as_index=False)["dollars_at_stake"].sum()
                 .pivot(index="verdict", columns="level", values="dollars_at_stake").fillna(0))
        st.bar_chart(pivot)
    else:
        st.info("No entities cleared the spend floor. Lower the minimum spend in the sidebar.")

# =================================================================== PACING ===
with tabs[1]:
    st.subheader("Budget pacing")
    if "budget" not in camp.columns or camp["budget"].isna().all():
        st.warning("No **Budget** column in the campaign export. Re-export with the Budget column "
                   "added — pacing cannot be computed without it.")
    else:
        pace = pacing_table(camp, days_in_data, days_in_month)
        pace = pace[pace["cost"] > 0] if pace["cost"].sum() > 0 else pace

        total_daily_budget = pace["budget"].sum()
        projected = pace["projected_month_spend"].sum()
        ceiling = total_daily_budget * MONTHLY_MULTIPLIER

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sum of daily budgets", money(total_daily_budget, sym))
        p2.metric("Theoretical monthly ceiling", money(ceiling, sym),
                  help="Google caps monthly spend at daily budget × 30.4, not × days in month.")
        p3.metric("Projected month-end spend", money(projected, sym))
        if monthly_cap > 0:
            over = projected - monthly_cap
            p4.metric("vs account cap", money(over, sym),
                      delta=f"{over/monthly_cap:+.0%}" if monthly_cap else None,
                      delta_color="inverse")

        if monthly_cap > 0 and ceiling > monthly_cap * 1.05:
            st.error(f"Sum of daily budgets could spend {money(ceiling, sym)} this month against a "
                     f"{money(monthly_cap, sym)} cap. If delivery improves, you breach the cap without "
                     "touching a single setting.")

        counts = pace["pacing_status"].value_counts()
        st.write(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))

        st.markdown("##### Capped and performing — raise these first")
        capped = pace[(pace["pacing_status"] == "Budget capped")]
        if use_roas:
            capped_good = capped[capped["roas"] >= target_roas].sort_values("conv_value", ascending=False)
        else:
            capped_good = capped[(capped["cpa"] <= target_cpa) & (capped["conversions"] > 0)].sort_values("conversions", ascending=False)
        if capped_good.empty:
            st.caption("Nothing is both budget-capped and clearing target. No free money here.")
        else:
            headroom = capped_good["cost"].sum() * 0.3
            st.success(f"{len(capped_good)} campaigns are hitting their budget ceiling while clearing target. "
                       f"A 30% budget lift on these puts roughly {money(headroom, sym)} more spend "
                       f"into proven ground.")
            cols = [c for c in ["campaign", "budget", "avg_daily_spend", "pace_vs_budget",
                                "lost_is_budget", "cost", "conv_value", "roas", "cpa",
                                "conversions", "projected_month_spend"] if c in capped_good.columns]
            st.dataframe(fmt_frame(capped_good[cols], sym), width="stretch", hide_index=True)
            dl_button(capped_good[cols], "⬇ Download scale list", "gads_scale_budgets.csv", "dl_scale")

        st.markdown("##### Capped and losing money — cap these before you raise anything")
        if use_roas:
            capped_bad = capped[(capped["roas"] < target_roas) | (capped["conversions"] == 0)]
        else:
            capped_bad = capped[(capped["cpa"] > target_cpa) | (capped["conversions"] == 0)]
        capped_bad = capped_bad.sort_values("cost", ascending=False)
        if capped_bad.empty:
            st.caption("No capped campaigns are underperforming.")
        else:
            cols = [c for c in ["campaign", "budget", "avg_daily_spend", "pace_vs_budget",
                                "cost", "conv_value", "roas", "cpa", "conversions"] if c in capped_bad.columns]
            st.dataframe(fmt_frame(capped_bad[cols], sym), width="stretch", hide_index=True)

        st.markdown("##### Full pacing table")
        cols = [c for c in ["campaign", "campaign_state", "pacing_status", "budget", "avg_daily_spend",
                            "pace_vs_budget", "projected_month_spend", "monthly_ceiling", "cost",
                            "conv_value", "roas", "cpa", "conversions", "lost_is_budget",
                            "lost_is_rank", "search_is"] if c in pace.columns]
        st.dataframe(fmt_frame(pace[cols].sort_values("cost", ascending=False), sym),
                     width="stretch", hide_index=True)
        dl_button(pace[cols], "⬇ Download pacing table", "gads_pacing.csv", "dl_pacing")

        if "date" in base.columns and base["date"].notna().any():
            st.markdown("##### Daily spend trend")
            daily = base.groupby(base["date"].dt.date, as_index=True)[["cost", "conv_value"]].sum()
            st.line_chart(daily)

# ================================================================ CAMPAIGNS ===
with tabs[2]:
    st.subheader("Campaign triage")
    cv = verdict(camp[camp["cost"] >= min_waste_cost], account_cvr, target_roas,
                 target_cpa, min_expected_conv, use_roas)
    pick_v = st.multiselect("Filter verdicts", sorted(cv["verdict"].unique()),
                            default=sorted(cv["verdict"].unique()), key="camp_v")
    show = cv[cv["verdict"].isin(pick_v)]
    cols = [c for c in ["campaign", "campaign_type", "bid_strategy", "verdict", "dollars_at_stake",
                        "cost", "conv_value", "roas", "cpa", "conversions", "clicks", "ctr",
                        "cvr", "cpc", "aov", "expected_conv"] if c in show.columns]
    st.dataframe(fmt_frame(show[cols], sym), width="stretch", hide_index=True)
    dl_button(show[cols], "⬇ Download campaign triage", "gads_campaigns.csv", "dl_camp")

    st.markdown("##### Spend vs ROAS — bubble size is spend")
    plot = show[show["cost"] > 0].copy()
    if not plot.empty and plot["roas"].notna().any():
        st.scatter_chart(plot, x="cost", y="roas", size="conv_value", color="verdict")
        st.caption(f"Anything below the {target_roas:.1f}x line is buying revenue you cannot afford.")

# ================================================================ AD GROUPS ===
with tabs[3]:
    st.subheader("Ad group triage")
    if "ad_groups" not in loaded:
        st.info("Upload an ad group export to use this tab.")
    else:
        ag = aggregate(loaded["ad_groups"], ["campaign", "ad_group", "ad_group_state"])
        ag = ag[ag["cost"] >= min_waste_cost]
        agv = verdict(ag, account_cvr, target_roas, target_cpa, min_expected_conv, use_roas)
        camps = st.multiselect("Limit to campaigns", sorted(agv["campaign"].dropna().unique().tolist()),
                               default=[], key="ag_camp")
        if camps:
            agv = agv[agv["campaign"].isin(camps)]
        cols = [c for c in ["campaign", "ad_group", "verdict", "dollars_at_stake", "cost",
                            "conv_value", "roas", "cpa", "conversions", "clicks", "ctr", "cvr",
                            "expected_conv"] if c in agv.columns]
        st.dataframe(fmt_frame(agv[cols], sym), width="stretch", hide_index=True)
        dl_button(agv[cols], "⬇ Download ad group triage", "gads_adgroups.csv", "dl_ag")

# ================================================================= KEYWORDS ===
with tabs[4]:
    st.subheader("Keyword triage")
    if "keywords" not in loaded:
        st.info("Upload a keyword export to use this tab.")
    else:
        kw_raw = loaded["keywords"]
        kw = aggregate(kw_raw, ["campaign", "ad_group", "keyword", "match_type"])
        kwv = verdict(kw[kw["cost"] >= min_waste_cost], account_cvr, target_roas,
                      target_cpa, min_expected_conv, use_roas)

        if "match_type" in kw.columns:
            st.markdown("##### Match-type economics")
            mt = aggregate(kw_raw, ["match_type"])
            cols = [c for c in ["match_type", "cost", "conv_value", "roas", "cpa", "conversions",
                                "clicks", "ctr", "cvr", "cpc"] if c in mt.columns]
            st.dataframe(fmt_frame(mt[cols].sort_values("cost", ascending=False), sym),
                         width="stretch", hide_index=True)
            st.caption("If broad match carries meaningful spend at below-target ROAS, that is a "
                       "structural leak — not a bid problem.")

        if "quality_score" in kw.columns and kw["quality_score"].notna().any():
            low_qs = kw[(kw["quality_score"] <= 4) & (kw["cost"] > min_waste_cost)]
            if not low_qs.empty:
                st.warning(f"{len(low_qs)} keywords with Quality Score ≤4 are spending "
                           f"{money(low_qs['cost'].sum(), sym)}. Low QS is a landing-page and "
                           "ad-relevance tax paid on every click.")

        st.markdown("##### Keyword action list")
        cols = [c for c in ["campaign", "ad_group", "keyword", "match_type", "verdict",
                            "dollars_at_stake", "cost", "conv_value", "roas", "cpa", "conversions",
                            "clicks", "cvr", "cpc", "quality_score", "expected_conv"] if c in kwv.columns]
        st.dataframe(fmt_frame(kwv[cols].head(500), sym), width="stretch", hide_index=True)
        dl_button(kwv[cols], "⬇ Download keyword triage", "gads_keywords.csv", "dl_kw")

# ============================================================ SEARCH TERMS ===
with tabs[5]:
    st.subheader("Search term mining")
    if "search_terms" not in loaded:
        st.info("Upload a search terms export to use this tab. This is the highest-yield "
                "tab in the app for a large account.")
    else:
        stf = loaded["search_terms"]
        st_agg = aggregate(stf, ["search_term"])

        st.markdown("##### N-gram waste — build account-level negative lists from these")
        st.caption("Individual search terms are too sparse to act on across hundreds of campaigns. "
                   "Recurring word patterns are not.")
        n = st.radio("N-gram size", [1, 2, 3], index=1, horizontal=True)
        gram = ngrams(stf, "search_term", n=n, min_cost=min_waste_cost)
        if gram.empty:
            st.caption("Not enough search term text to build n-grams at this spend floor.")
        else:
            gram["expected_conv"] = gram["clicks"] * max(account_cvr, 1e-9)
            waste = gram[(gram["conversions"] == 0) & (gram["expected_conv"] >= min_expected_conv)]
            win = gram[(gram["conversions"] > 0)]
            if use_roas:
                win = win[win["roas"] >= target_roas]
            else:
                win = win[win["cpa"] <= target_cpa]

            w1, w2 = st.columns(2)
            with w1:
                st.markdown(f"**Negative candidates — {money(waste['cost'].sum(), sym)} at stake**")
                cols = [c for c in ["ngram", "cost", "clicks", "conversions", "terms", "expected_conv"] if c in waste.columns]
                st.dataframe(fmt_frame(waste[cols].head(150), sym), width="stretch", hide_index=True)
                dl_button(waste[cols], "⬇ Download negative candidates", "gads_negative_ngrams.csv", "dl_neg")
            with w2:
                st.markdown(f"**Winning patterns — {money(win['conv_value'].sum(), sym)} revenue**")
                cols = [c for c in ["ngram", "cost", "conv_value", "roas", "cpa", "conversions", "terms"] if c in win.columns]
                st.dataframe(fmt_frame(win[cols].head(150), sym), width="stretch", hide_index=True)
                dl_button(win[cols], "⬇ Download winning patterns", "gads_winning_ngrams.csv", "dl_win")

        st.markdown("##### Individual terms")
        stv = verdict(st_agg[st_agg["cost"] >= min_waste_cost], account_cvr, target_roas,
                      target_cpa, min_expected_conv, use_roas)
        if "added_excluded" in stf.columns:
            st.caption("Terms already added or excluded are still shown — check the source export "
                       "before creating duplicates.")
        cols = [c for c in ["search_term", "verdict", "dollars_at_stake", "cost", "conv_value",
                            "roas", "cpa", "conversions", "clicks", "cvr", "expected_conv"] if c in stv.columns]
        st.dataframe(fmt_frame(stv[cols].head(500), sym), width="stretch", hide_index=True)
        dl_button(stv[cols], "⬇ Download search term triage", "gads_search_terms.csv", "dl_st")

# ====================================================================== ADS ===
with tabs[6]:
    st.subheader("Creative triage")
    if "ads" not in loaded:
        st.info("Upload an ad-level export to use this tab.")
    else:
        adf = loaded["ads"]
        keys = [k for k in ["campaign", "ad_group", "ad_label", "final_url"] if k in adf.columns]
        ads = aggregate(adf, keys)
        ads = ads[ads["cost"] >= min_waste_cost]
        if ads.empty:
            st.caption("No ads above the spend floor.")
        else:
            grp = [k for k in ["campaign", "ad_group"] if k in ads.columns]
            if grp:
                ads["ag_ctr"] = ads.groupby(grp)["ctr"].transform("mean")
                ads["ag_cvr"] = ads.groupby(grp)["cvr"].transform("mean")
                ads["ctr_index"] = ads["ctr"] / ads["ag_ctr"]
                ads["cvr_index"] = ads["cvr"] / ads["ag_cvr"]
            adv = verdict(ads, account_cvr, target_roas, target_cpa, min_expected_conv, use_roas)
            cols = [c for c in ["campaign", "ad_group", "ad_label", "verdict", "dollars_at_stake",
                                "cost", "conv_value", "roas", "conversions", "clicks", "ctr",
                                "ctr_index", "cvr", "cvr_index"] if c in adv.columns]
            show = adv[cols].copy()
            for c in ("ctr_index", "cvr_index"):
                if c in show.columns:
                    show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
            st.dataframe(fmt_frame(show.head(400), sym), width="stretch", hide_index=True)
            st.caption("Index below 1.00 means the ad underperforms its own ad group average. "
                       "Pause only where a sibling ad has enough data to take the traffic.")
            dl_button(adv[cols], "⬇ Download ad triage", "gads_ads.csv", "dl_ads")

# =============================================================== DATA CHECK ===
with tabs[7]:
    st.subheader("Data check")
    st.caption("Read this before trusting anything above.")
    for key, df in loaded.items():
        with st.expander(f"{key} — {len(df):,} rows, {df.shape[1]} columns"):
            recognised = [c for c in df.columns if c in CANON]
            unrecognised = [c for c in df.columns if c not in CANON]
            st.write("**Mapped:**", ", ".join(recognised) or "none")
            st.write("**Unmapped (ignored):**", ", ".join(map(str, unrecognised)) or "none")
            missing = [c for c in ["cost", "clicks", "conversions", "conv_value"]
                       if c not in df.columns or df[c].sum() == 0]
            if missing:
                st.warning(f"Zero or missing: {', '.join(missing)}. If those columns exist in your "
                           "export under a different name, they were not recognised — check the "
                           "unmapped list above.")
            st.dataframe(df.head(8), width="stretch")

    st.markdown("---")
    st.markdown(
        """
**Known limits of this tool — do not skip these.**

1. **Revenue here is Google-reported, not GA4 and not your order system.** Platform conversion
   value is credited on Google's own attribution window and will not reconcile with backend
   revenue. Treat ROAS in this app as a *relative ranking signal between campaigns*, not as a
   true return figure.
2. **Conversion value is not summable across platforms.** Never add this to Meta or GA4 numbers.
3. **Blank conversion columns usually mean a tracking fault, not a performance fault.** If a
   whole campaign type shows zero conversions, verify tracking before cutting anything.
4. **The guardrail is a floor, not proof.** "Not enough data" means exactly that — do not read it
   as "fine".
5. **PMax and Shopping do not expose search terms at keyword level.** The search-term tab covers
   Search campaigns; PMax waste has to be found in the campaign-level view and asset group reports.
        """
    )
