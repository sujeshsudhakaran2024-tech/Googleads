"""
Cables on Demand — Google Ads Pacing Monitor (Streamlit)

Replicates COD_Pacing_Monitor.xlsx and fixes two flaws in it:
  1. Schedule basis is per-campaign (7-day vs weekday), so PMax campaigns
     stop being flagged "needs attention" by construction.
  2. Elapsed-days denominator matches the schedule basis, so the run-rate
     projection is not distorted by which weekday the month started on.

Also adds revenue pacing, because hitting a spend target at a bad ROAS
is not a result worth a green light.

Run:  streamlit run cod_pacing_app.py
Deps: streamlit pandas numpy plotly openpyxl
"""

import calendar
import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

BILLABLE_MULTIPLIER = 30.4  # Google's monthly billing cap = daily budget x 30.4

BRAND_BLUE = "#005CB9"
BRAND_CYAN = "#00CADB"
BRAND_NAVY = "#03045E"

STATUS_COLORS = {
    "No budget set": "#6A1B9A",
    "Over plan": "#C62828",
    "At cap": "#EF6C00",
    "On plan": "#2E7D32",
    "Under plan": "#1565C0",
}

# Seeded with Aug 1-12 2026 actuals from the Google Ads campaign export so
# spend and revenue come from the same window. Overwrite via CSV upload.
DEFAULT_CAMPAIGNS = pd.DataFrame(
    [
        {"Campaign": "SC - TM - Cables On Demand",            "Daily Budget": 35.0,  "Basis": "7-day", "Actual Spend MTD": 195.84,  "Actual Revenue MTD": 2576.92},
        {"Campaign": "SC - PN - Branded Part Numbers",         "Daily Budget": 25.0,  "Basis": "7-day", "Actual Spend MTD": 63.16,   "Actual Revenue MTD": 146.83},
        {"Campaign": "QT - PMAX - Priority Products",          "Daily Budget": 87.5,  "Basis": "7-day", "Actual Spend MTD": 1061.65, "Actual Revenue MTD": 1886.31},
        {"Campaign": "QT - PMAX - Under Performing Products",  "Daily Budget": 37.5,  "Basis": "7-day", "Actual Spend MTD": 421.64,  "Actual Revenue MTD": 3101.41},
    ]
)


# ----------------------------------------------------------------------------
# Calendar helpers
# ----------------------------------------------------------------------------

def month_bounds(year: int, month: int):
    days = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), days


def weekdays_in_month(year: int, month: int) -> int:
    days = calendar.monthrange(year, month)[1]
    return sum(1 for d in range(1, days + 1) if dt.date(year, month, d).weekday() < 5)


def weekdays_elapsed(year: int, month: int, days_elapsed: int) -> int:
    return sum(1 for d in range(1, days_elapsed + 1) if dt.date(year, month, d).weekday() < 5)


# ----------------------------------------------------------------------------
# Core pacing model
# ----------------------------------------------------------------------------

def compute_pacing(df: pd.DataFrame, year: int, month: int, days_elapsed: int,
                   threshold: float) -> pd.DataFrame:
    """Per-campaign pacing. Denominator matches each row's schedule basis."""
    cal_days = calendar.monthrange(year, month)[1]
    wd_total = weekdays_in_month(year, month)
    wd_elapsed = weekdays_elapsed(year, month, days_elapsed)

    out = df.copy()

    is_weekday = out["Basis"].eq("Weekday")
    # An always-on campaign is planned against Google's billable month (30.4x),
    # not the raw calendar day count -- you can never be charged more than that.
    sched_days = np.where(is_weekday, wd_total, BILLABLE_MULTIPLIER)
    elapsed = np.where(is_weekday, wd_elapsed, days_elapsed)
    # Run-rate always projects across the real calendar remaining.
    project_days = np.where(is_weekday, wd_total, cal_days)

    out["Sched Days"] = sched_days
    out["Days Elapsed"] = elapsed

    # Plan uses the same basis the campaign actually runs on.
    out["Planned Monthly"] = out["Daily Budget"] * sched_days
    out["Monthly Cap"] = out["Daily Budget"] * BILLABLE_MULTIPLIER

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_proj = np.where(elapsed > 0,
                            out["Actual Spend MTD"] / np.maximum(elapsed, 1) * project_days,
                            np.nan)

    # You cannot be billed above the monthly cap, so the projection is capped.
    out["Raw Projection"] = raw_proj
    out["Projected Spend"] = np.minimum(raw_proj, out["Monthly Cap"])

    out["Variance"] = (out["Projected Spend"] - out["Planned Monthly"]) / out["Planned Monthly"].replace(0, np.nan)
    out["At Cap"] = out["Raw Projection"] >= out["Monthly Cap"] * 0.999

    # Revenue side
    out["Projected Revenue"] = np.where(elapsed > 0,
                                        out["Actual Revenue MTD"] / np.maximum(elapsed, 1) * project_days,
                                        np.nan)
    out["ROAS MTD"] = out["Actual Revenue MTD"] / out["Actual Spend MTD"].replace(0, np.nan)
    out["Spend Pace %"] = out["Actual Spend MTD"] / (out["Daily Budget"] * elapsed).replace(0, np.nan)

    def status(row):
        if not row["Daily Budget"] or row["Daily Budget"] <= 0:
            return "No budget set"
        if pd.isna(row["Variance"]):
            return "No budget set"
        if row["At Cap"]:
            return "At cap"
        if row["Variance"] > threshold:
            return "Over plan"
        if row["Variance"] < -threshold:
            return "Under plan"
        return "On plan"

    out["Status"] = out.apply(status, axis=1)
    return out


def build_delivery_curve(df: pd.DataFrame, year: int, month: int, days_elapsed: int):
    """Cumulative account spend: ideal straight-line plan vs implied actual."""
    cal_days = calendar.monthrange(year, month)[1]
    days = np.arange(1, cal_days + 1)

    planned_total = df["Planned Monthly"].sum()
    actual_total = df["Actual Spend MTD"].sum()
    projected_total = df["Projected Spend"].sum()

    plan_line = planned_total * days / cal_days
    actual_line = np.where(days <= days_elapsed, actual_total * days / max(days_elapsed, 1), np.nan)
    proj_line = np.where(days >= days_elapsed,
                         actual_total + (projected_total - actual_total)
                         * (days - days_elapsed) / max(cal_days - days_elapsed, 1),
                         np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=plan_line, name="Plan",
                             line=dict(color="#9E9E9E", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=days, y=actual_line, name="Actual MTD",
                             line=dict(color=BRAND_BLUE, width=3)))
    fig.add_trace(go.Scatter(x=days, y=proj_line, name="Projected",
                             line=dict(color=BRAND_CYAN, width=2, dash="dot")))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Day of month", yaxis_title="Cumulative spend ($)",
        yaxis_tickprefix="$", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


# ----------------------------------------------------------------------------
# Google Ads CSV ingestion
# ----------------------------------------------------------------------------

def _read_ads_csv(raw: bytes) -> pd.DataFrame:
    """Google Ads exports vary by encoding, delimiter, and number of preamble rows."""
    combos = [("utf-8", ","), ("utf-16", "\t"), ("utf-16", ","),
              ("utf-8-sig", ","), ("latin-1", ","), ("utf-8", "\t")]
    for enc, sep in combos:
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        lines = text.splitlines()
        # Find the real header row rather than assuming a fixed preamble depth.
        for skip in range(0, min(8, len(lines))):
            if "campaign" not in lines[skip].lower():
                continue
            try:
                cand = pd.read_csv(io.StringIO(text), skiprows=skip, sep=sep,
                                   thousands=",", keep_default_na=False)
            except Exception:
                continue
            cols = [c.strip() for c in cand.columns]
            if "Campaign" in cols and len(cols) > 3:
                cand.columns = cols
                return cand
    raise ValueError(
        "Could not read this file. Export the Campaigns table from Google Ads "
        "as CSV without editing it, and make sure Cost is one of the columns."
    )


def parse_google_ads_export(uploaded) -> pd.DataFrame:
    """Pull campaign, daily budget, cost and conversion value from any Google Ads
    campaign export. Works across accounts -- nothing is hardcoded to COD."""
    df = _read_ads_csv(uploaded.getvalue())

    def find(*, exact=None, contains=None):
        for c in df.columns:
            low = c.lower().strip()
            if any(t in low for t in ("compare to", "change")):
                continue
            if exact and low == exact:
                return c
            if contains and all(t in low for t in contains) and "/" not in low:
                return c
        return None

    cost_col = find(exact="cost")
    if cost_col is None:
        raise ValueError("No 'Cost' column in this export. Add it in the Google Ads "
                         "column picker and download again.")

    rev_col = (find(contains=["conv. value", "conv. time"])
               or find(exact="conv. value")
               or find(contains=["conv. value"])
               or find(contains=["all conv. value"]))
    budget_col = find(exact="budget") or find(contains=["daily budget"])
    status_col = find(exact="campaign status") or find(exact="campaign state")
    cur_col = find(exact="currency code")
    type_col = find(exact="budget type")

    names = df["Campaign"].astype(str).str.strip()
    df = df[(~names.str.startswith(("Total", "--", "-"))) & (names != "")].copy()

    period_col = next((c for c in df.columns
                       if c.strip() in ("Month", "Day", "Week", "Quarter", "Year")), None)
    multi_period = bool(period_col) and df[period_col].astype(str).str.strip().nunique() > 1

    dropped_removed = 0
    if status_col:
        st_vals = df[status_col].astype(str).str.strip().str.lower()
        keep = ~st_vals.isin(["removed", "paused"])
        dropped_removed = int((~keep).sum())
        if keep.any():
            df = df[keep]

    def num(sr):
        return pd.to_numeric(
            sr.astype(str)
              .str.replace(r"[^\d.\-]", "", regex=True)
              .replace("", "0"),
            errors="coerce").fillna(0.0)

    df[cost_col] = num(df[cost_col])
    if rev_col:
        df[rev_col] = num(df[rev_col])
    if budget_col:
        df[budget_col] = num(df[budget_col])

    agg = {"Actual Spend MTD": (cost_col, "sum")}
    if rev_col:
        agg["Actual Revenue MTD"] = (rev_col, "sum")
    if budget_col:
        # Budget repeats per row in a monthly breakdown -- take the latest value.
        agg["Daily Budget"] = (budget_col, "max")

    g = df.groupby("Campaign", as_index=False).agg(**agg)
    for col, default in (("Actual Revenue MTD", 0.0), ("Daily Budget", 0.0)):
        if col not in g.columns:
            g[col] = default

    currencies = (sorted(df[cur_col].astype(str).str.strip().unique()) if cur_col else [])
    currencies = [c for c in currencies if c and c not in ("--", "-")]

    non_daily = []
    if type_col:
        non_daily = sorted({str(v).strip() for v in df[type_col]
                            if str(v).strip().lower() not in ("daily", "", "--", "-")})

    g.attrs.update(
        multi_period=multi_period,
        budget_found=bool(budget_col),
        revenue_found=bool(rev_col),
        currencies=currencies,
        dropped_removed=dropped_removed,
        non_daily=non_daily,
    )
    return g


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="COD Pacing Monitor", page_icon="📊", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; }}
      h1 {{ color: {BRAND_NAVY}; font-size: 1.9rem !important; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Google Ads Pacing Monitor — Cables on Demand")

# --- Sidebar: period + thresholds ---
with st.sidebar:
    st.header("Reporting period")
    today = dt.date.today()
    period = st.date_input("Month to monitor", value=today.replace(day=1))
    year, month = period.year, period.month
    cal_days = calendar.monthrange(year, month)[1]

    auto = st.checkbox("Use today's date for days elapsed", value=True)
    if auto and (today.year, today.month) == (year, month):
        days_elapsed = today.day
    else:
        days_elapsed = st.slider("Calendar days elapsed", 1, cal_days,
                                 min(today.day, cal_days))
    if auto and (today.year, today.month) != (year, month):
        days_elapsed = cal_days
        st.caption("Past month selected — treating it as complete.")

    st.metric("Days elapsed", f"{days_elapsed} / {cal_days}")

    threshold = st.slider("Attention threshold (± vs plan)", 0.05, 0.30, 0.10, 0.01,
                          format="%.0f%%")
    st.caption("Nick's rule: flag anything projecting more than this above plan.")

    st.divider()
    st.header("Load actuals")
    up = st.file_uploader("Google Ads campaign report (.csv)", type=["csv"])
    st.caption("Google Ads → Campaigns → Download → CSV. "
               "Set the date range to month-to-date before exporting.")

# --- Campaign inputs ---
if "campaigns" not in st.session_state:
    st.session_state.campaigns = DEFAULT_CAMPAIGNS.copy()

if up is not None:
    try:
        parsed = parse_google_ads_export(up)
        base = st.session_state.campaigns.set_index("Campaign")
        merged = []
        for _, r in parsed.iterrows():
            name = r["Campaign"]
            prior = base.loc[name] if name in base.index else None
            budget = float(r.get("Daily Budget", 0.0) or 0.0)
            if budget <= 0 and prior is not None:
                budget = float(prior["Daily Budget"])
            merged.append({
                "Campaign": name,
                "Daily Budget": budget,
                "Basis": prior["Basis"] if prior is not None else "7-day",
                "Actual Spend MTD": float(r["Actual Spend MTD"]),
                "Actual Revenue MTD": float(r["Actual Revenue MTD"]),
            })
        st.session_state.campaigns = pd.DataFrame(merged)

        if parsed.attrs.get("currencies") and len(parsed.attrs["currencies"]) > 1:
            st.sidebar.error(
                "This export mixes currencies (" + ", ".join(parsed.attrs["currencies"])
                + "). Totals across campaigns are not comparable -- export one account "
                  "at a time."
            )
        elif parsed.attrs.get("currencies"):
            st.session_state.currency = parsed.attrs["currencies"][0]
        if not parsed.attrs.get("budget_found"):
            st.sidebar.warning(
                "No Budget column in this export -- enter daily budgets by hand, or "
                "add Budget in the Google Ads column picker and re-download."
            )
        if not parsed.attrs.get("revenue_found"):
            st.sidebar.warning(
                "No conversion value column found. Spend pacing will work; the "
                "return panel will not."
            )
        if parsed.attrs.get("non_daily"):
            st.sidebar.warning(
                "Non-daily budget types present (" + ", ".join(parsed.attrs["non_daily"])
                + "). This model assumes daily budgets."
            )
        if parsed.attrs.get("dropped_removed"):
            st.sidebar.caption(
                f"Excluded {parsed.attrs['dropped_removed']} paused/removed campaign rows."
            )
        if parsed.attrs.get("multi_period"):
            st.sidebar.warning(
                "This export covers more than one period, so the figures below are a "
                "sum of every month in it — not month-to-date. Re-export with the date "
                "range set to this month only."
            )
        st.sidebar.success(f"Loaded {len(merged)} campaigns. Confirm daily budgets below.")
    except Exception as e:
        st.sidebar.error(str(e))

st.subheader("Campaign inputs")
st.caption("Daily budget and schedule basis come from Google Ads settings. "
           "Spend and revenue auto-fill from the CSV upload, or type them in.")

edited = st.data_editor(
    st.session_state.campaigns,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Campaign": st.column_config.TextColumn(width="large"),
        "Daily Budget": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
        "Basis": st.column_config.SelectboxColumn(
            options=["7-day", "Weekday"],
            help="7-day = campaign delivers every day (PMax, always-on Search). "
                 "Weekday = ad schedule restricts it to Mon-Fri.",
        ),
        "Actual Spend MTD": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
        "Actual Revenue MTD": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
    },
    key="editor",
)
st.session_state.campaigns = edited

work = edited.dropna(subset=["Campaign"])
work = work[work["Campaign"].astype(str).str.strip() != ""]

if work.empty:
    st.info("Add at least one campaign to see pacing.")
    st.stop()

paced = compute_pacing(work, year, month, days_elapsed, threshold)

# --- Account summary ---
st.divider()
budgeted = paced[paced["Daily Budget"] > 0]
acct_planned = budgeted["Planned Monthly"].sum()
acct_actual = budgeted["Actual Spend MTD"].sum()
acct_proj = budgeted["Projected Spend"].sum()
acct_rev_proj = budgeted["Projected Revenue"].sum()
acct_var = (acct_proj - acct_planned) / acct_planned if acct_planned else np.nan
acct_roas = budgeted["Actual Revenue MTD"].sum() / acct_actual if acct_actual else np.nan
proj_roas = acct_rev_proj / acct_proj if acct_proj else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spend MTD", f"${acct_actual:,.0f}", f"{acct_actual/acct_planned:.0%} of plan"
          if acct_planned else None)
c2.metric("Projected month-end", f"${acct_proj:,.0f}",
          f"{acct_var:+.0%} vs plan" if pd.notna(acct_var) else None)
c3.metric("Planned monthly", f"${acct_planned:,.0f}")
c4.metric("ROAS MTD", f"{acct_roas:.2f}x" if pd.notna(acct_roas) else "—")
c5.metric("Projected revenue", f"${acct_rev_proj:,.0f}",
          f"{proj_roas:.2f}x projected ROAS" if pd.notna(proj_roas) else None)

unbudgeted = paced[paced["Daily Budget"] <= 0]
if not unbudgeted.empty:
    st.error(
        "**No daily budget set for: " + ", ".join(unbudgeted["Campaign"].astype(str))
        + f"** -- ${unbudgeted['Actual Spend MTD'].sum():,.0f} of spend is excluded "
          "from every plan and projection below. Fill in the budgets above."
    )

if pd.notna(acct_var):
    if acct_var > threshold:
        st.error(f"Account is pacing **{acct_var:+.0%} above plan**. "
                 f"Projected overspend: ${acct_proj - acct_planned:,.0f}.")
    elif acct_var < -threshold:
        st.warning(f"Account is pacing **{acct_var:+.0%} below plan**. "
                   f"${acct_planned - acct_proj:,.0f} of budget will go unspent.")
    else:
        st.success(f"Account on plan (within {threshold:.0%}).")

# --- Detail table ---
st.subheader("Campaign detail")

display = paced[[
    "Campaign", "Daily Budget", "Basis", "Planned Monthly", "Monthly Cap",
    "Actual Spend MTD", "Projected Spend", "Variance", "ROAS MTD", "Status",
]].copy()


def style_row(row):
    color = STATUS_COLORS.get(row["Status"], "#000000")
    return [f"color: {color}; font-weight: 600" if col == "Status" else ""
            for col in row.index]


st.dataframe(
    display.style
    .apply(style_row, axis=1)
    .format({
        "Daily Budget": "${:,.2f}",
        "Planned Monthly": "${:,.0f}",
        "Monthly Cap": "${:,.0f}",
        "Actual Spend MTD": "${:,.2f}",
        "Projected Spend": "${:,.0f}",
        "Variance": "{:+.1%}",
        "ROAS MTD": "{:.2f}x",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

# --- Charts ---
left, right = st.columns([3, 2])

with left:
    st.markdown("**Account delivery curve**")
    st.plotly_chart(build_delivery_curve(paced, year, month, days_elapsed),
                    use_container_width=True)

with right:
    st.markdown("**Variance vs plan**")
    v = paced.sort_values("Variance")
    fig = go.Figure(go.Bar(
        x=v["Variance"] * 100,
        y=v["Campaign"].str.replace(" - ", "\n", regex=False),
        orientation="h",
        marker_color=[STATUS_COLORS.get(s, "#9E9E9E") for s in v["Status"]],
        text=[f"{x:+.0%}" for x in v["Variance"]],
        textposition="outside",
    ))
    fig.add_vline(x=threshold * 100, line_dash="dot", line_color="#C62828")
    fig.add_vline(x=-threshold * 100, line_dash="dot", line_color="#1565C0")
    fig.update_layout(height=340, margin=dict(l=10, r=40, t=30, b=10),
                      xaxis_title="Variance vs plan (%)", plot_bgcolor="white",
                      showlegend=False)
    fig.update_xaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

# --- Revenue efficiency: the part the spreadsheet omits ---
st.divider()
st.subheader("Spend pacing vs return")
st.caption("Pacing to budget is only good news if the money is working. "
           "Bubble size = projected month-end spend.")

eff = paced[paced["Actual Spend MTD"] > 0]
if not eff.empty:
    fig = go.Figure(go.Scatter(
        x=eff["Variance"] * 100,
        y=eff["ROAS MTD"],
        mode="markers+text",
        text=eff["Campaign"],
        textposition="top center",
        marker=dict(
            size=eff["Projected Spend"] / max(eff["Projected Spend"].max(), 1) * 70 + 12,
            color=[STATUS_COLORS.get(s, "#9E9E9E") for s in eff["Status"]],
            opacity=0.75, line=dict(width=1, color="white"),
        ),
    ))
    target = st.number_input("Target ROAS", min_value=0.5, max_value=20.0,
                             value=4.5, step=0.5)
    fig.add_hline(y=target, line_dash="dash", line_color="#2E7D32",
                  annotation_text=f"Target {target:.1f}x")
    fig.add_vline(x=0, line_dash="dot", line_color="#9E9E9E")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Variance vs plan (%)", yaxis_title="ROAS MTD",
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Read it this way.** Bottom-right is the expensive quadrant: spending "
        "ahead of plan below target ROAS. Top-left is the missed opportunity: "
        "earning well but underspending. Fix the bottom-right first, then feed "
        "the top-left."
    )

# --- Export ---
st.divider()
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xl:
    paced.to_excel(xl, sheet_name="Pacing", index=False)
st.download_button("Download this view as Excel", buf.getvalue(),
                   file_name=f"COD_Pacing_{year}-{month:02d}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("How the numbers are calculated"):
    st.markdown(f"""
- **Planned Monthly** = daily budget x scheduled days. A **7-day** campaign is planned
  against Google's billable month ({BILLABLE_MULTIPLIER} days); a **Weekday** campaign
  against the **{weekdays_in_month(year, month)}** weekdays in this month.
- **Monthly Cap** = daily budget x {BILLABLE_MULTIPLIER}. Google's billing ceiling —
  you cannot be charged above it, so the projection is capped there. For always-on
  campaigns the plan *is* the cap, which means they can only ever pace **under** plan.
  Overspend risk exists only where an ad schedule restricts delivery.
- **Projected Spend** = MTD spend / elapsed days x scheduled days, capped at the
  monthly cap. Elapsed days uses the same basis as the plan
  (**{days_elapsed}** calendar / **{weekdays_elapsed(year, month, days_elapsed)}** weekday).
- **Variance** = (Projected - Planned) / Planned.
- **Status** flags at ±{threshold:.0%}. "At cap" means the raw run-rate exceeds the
  billing ceiling — the campaign is demand-limited, not budget-limited.

**Why the basis matters.** The Excel version planned every campaign against 22
weekdays but projected against calendar days. That built a permanent +38%
({BILLABLE_MULTIPLIER}/22) overspend flag into both PMax campaigns, then explained it
away in a footnote. Setting PMax to a 7-day basis makes the flag mean something again.
""")
