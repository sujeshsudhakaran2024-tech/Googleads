"""
Google Ads Pacing Monitor
=========================

Budget pacing for any Google Ads account, across business units, for both
ecommerce (revenue / ROAS) and lead-gen (conversions / CPA) objectives.

Nothing in here is tied to a specific advertiser. Point it at any campaign
export.

Design notes
------------
* An always-on campaign is planned against Google's billable month
  (daily budget x 30.4), not the raw calendar-day count. Planning it against
  weekday counts while projecting against calendar days builds a permanent
  ~38% false overspend flag into every always-on campaign.
* Elapsed days always use the same basis as the plan.
* All model inputs are sanitised before use. A single cleared cell in the
  editor must never take the page down.

Run:  streamlit run pacing_monitor.py
Deps: streamlit pandas numpy plotly openpyxl
"""

import calendar
import datetime as dt
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BILLABLE_MULTIPLIER = 30.4  # Google's monthly billing ceiling

INK = "#12263A"
ACCENT = "#1E6FD9"
ACCENT_2 = "#00B8C4"

STATUS_COLORS = {
    "No budget set": "#6A1B9A",
    "Awaiting data": "#757575",
    "Over plan": "#C62828",
    "At cap": "#EF6C00",
    "On plan": "#2E7D32",
    "Under plan": "#1565C0",
}

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "INR": "\u20b9",
    "JPY": "\u00a5", "CNY": "\u00a5", "AUD": "A$", "CAD": "C$",
    "SGD": "S$", "CHF": "CHF ", "MXN": "MX$", "BRL": "R$",
}

OBJECTIVES = ["Revenue", "Leads"]

EDITOR_COLUMNS = [
    "Business Unit", "Campaign", "Objective", "Daily Budget", "Basis",
    "Actual Spend MTD", "Conversions MTD", "Conv. Value MTD",
]

SAMPLE_ROWS = pd.DataFrame([
    {"Business Unit": "BU A", "Campaign": "Brand - Exact",     "Objective": "Revenue",
     "Daily Budget": 35.0, "Basis": "7-day", "Actual Spend MTD": 196.0,
     "Conversions MTD": 8.0, "Conv. Value MTD": 2577.0},
    {"Business Unit": "BU A", "Campaign": "Shopping - All",    "Objective": "Revenue",
     "Daily Budget": 87.5, "Basis": "7-day", "Actual Spend MTD": 1062.0,
     "Conversions MTD": 12.0, "Conv. Value MTD": 1886.0},
    {"Business Unit": "BU B", "Campaign": "Generic - Quotes",  "Objective": "Leads",
     "Daily Budget": 60.0, "Basis": "Weekday", "Actual Spend MTD": 430.0,
     "Conversions MTD": 9.0, "Conv. Value MTD": 0.0},
])


def money(value, symbol="$", decimals=0):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "\u2014"
    return f"{symbol}{value:,.{decimals}f}"


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "account"


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def weekdays_in_month(year: int, month: int) -> int:
    days = calendar.monthrange(year, month)[1]
    return sum(1 for d in range(1, days + 1) if dt.date(year, month, d).weekday() < 5)


def weekdays_elapsed(year: int, month: int, days_elapsed: int) -> int:
    return sum(1 for d in range(1, int(days_elapsed) + 1)
               if dt.date(year, month, d).weekday() < 5)


# ---------------------------------------------------------------------------
# Pacing model
# ---------------------------------------------------------------------------

def compute_pacing(df: pd.DataFrame, year: int, month: int, days_elapsed: int,
                   threshold: float) -> pd.DataFrame:
    cal_days = calendar.monthrange(year, month)[1]
    wd_total = weekdays_in_month(year, month)
    wd_elapsed = weekdays_elapsed(year, month, days_elapsed)

    out = df.copy()

    # Sanitise every model input. The editor returns None for cleared cells and
    # blank rows for anything half-added; unchecked, those NaNs reach Plotly
    # marker sizes, which reject them and crash the page.
    for col in ("Daily Budget", "Actual Spend MTD", "Conversions MTD", "Conv. Value MTD"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    for col, default in (("Basis", "7-day"), ("Objective", "Revenue"),
                         ("Business Unit", "Unassigned")):
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).replace("", default)

    is_weekday = out["Basis"].eq("Weekday")
    # Always-on campaigns are planned against the billable month, not raw
    # calendar days -- you can never be charged more than 30.4x daily budget.
    sched_days = np.where(is_weekday, wd_total, BILLABLE_MULTIPLIER)
    elapsed = np.where(is_weekday, wd_elapsed, days_elapsed).astype(float)
    project_days = np.where(is_weekday, wd_total, cal_days).astype(float)

    out["Sched Days"] = sched_days
    out["Days Elapsed"] = elapsed

    out["Planned Monthly"] = out["Daily Budget"] * sched_days
    out["Monthly Cap"] = out["Daily Budget"] * BILLABLE_MULTIPLIER

    safe_elapsed = np.where(elapsed > 0, elapsed, np.nan)

    def project(series):
        return series.to_numpy(dtype=float) / safe_elapsed * project_days

    out["Raw Projection"] = project(out["Actual Spend MTD"])
    out["Projected Spend"] = np.minimum(out["Raw Projection"], out["Monthly Cap"])
    out["Projected Conversions"] = project(out["Conversions MTD"])
    out["Projected Value"] = project(out["Conv. Value MTD"])

    planned = out["Planned Monthly"].replace(0, np.nan)
    out["Variance"] = (out["Projected Spend"] - out["Planned Monthly"]) / planned
    out["At Cap"] = out["Raw Projection"] >= out["Monthly Cap"] * 0.999

    spend = out["Actual Spend MTD"].replace(0, np.nan)
    convs = out["Conversions MTD"].replace(0, np.nan)
    # A lead-gen campaign carries no conversion value, so a 0.00x ROAS would be
    # noise rather than a finding -- blank it instead.
    out["ROAS MTD"] = np.where(out["Conv. Value MTD"] > 0,
                               out["Conv. Value MTD"] / spend, np.nan)
    out["CPA MTD"] = out["Actual Spend MTD"] / convs
    # One efficiency column so charts and tables can stay objective-agnostic.
    out["Efficiency"] = np.where(out["Objective"].eq("Leads"),
                                 out["CPA MTD"], out["ROAS MTD"])

    def status(row):
        if not row["Daily Budget"] or row["Daily Budget"] <= 0:
            return "No budget set"
        # A weekday-basis campaign in a month that opens on a weekend has no
        # scheduled days elapsed yet -- nothing to project from.
        if row["Days Elapsed"] <= 0 or pd.isna(row["Variance"]):
            return "Awaiting data"
        if row["At Cap"]:
            return "At cap"
        if row["Variance"] > threshold:
            return "Over plan"
        if row["Variance"] < -threshold:
            return "Under plan"
        return "On plan"

    out["Status"] = out.apply(status, axis=1)
    return out


def bubble_sizes(values, floor: float = 12.0, span: float = 70.0):
    """Plotly rejects NaN and negative marker sizes outright."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).clip(lower=0.0)
    peak = v.max()
    if not np.isfinite(peak) or peak <= 0:
        return [floor] * len(v)
    return (v / peak * span + floor).tolist()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def delivery_curve(df: pd.DataFrame, year: int, month: int, days_elapsed: int, sym: str):
    cal_days = calendar.monthrange(year, month)[1]
    days = np.arange(1, cal_days + 1)

    planned = float(df["Planned Monthly"].sum())
    actual = float(df["Actual Spend MTD"].sum())
    projected = float(df["Projected Spend"].sum(skipna=True))
    elapsed = max(int(days_elapsed), 1)

    plan_line = planned * days / cal_days
    actual_line = np.where(days <= elapsed, actual * days / elapsed, np.nan)
    remaining = max(cal_days - elapsed, 1)
    proj_line = np.where(days >= elapsed,
                         actual + (projected - actual) * (days - elapsed) / remaining,
                         np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=plan_line, name="Plan",
                             line=dict(color="#9AA5B1", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=days, y=actual_line, name="Actual MTD",
                             line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=days, y=proj_line, name="Projected",
                             line=dict(color=ACCENT_2, width=2, dash="dot")))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Day of month",
                      yaxis_title=f"Cumulative spend ({sym})",
                      yaxis_tickprefix=sym, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#EEF1F4")
    fig.update_yaxes(gridcolor="#EEF1F4")
    return fig


def variance_bars(df: pd.DataFrame, threshold: float):
    v = df.sort_values("Variance", na_position="first")
    fig = go.Figure(go.Bar(
        x=(v["Variance"] * 100).fillna(0),
        y=v["Campaign"].astype(str),
        orientation="h",
        marker_color=[STATUS_COLORS.get(s, "#9AA5B1") for s in v["Status"]],
        text=["\u2014" if pd.isna(x) else f"{x:+.0%}" for x in v["Variance"]],
        textposition="outside",
    ))
    fig.add_vline(x=threshold * 100, line_dash="dot", line_color="#C62828")
    fig.add_vline(x=-threshold * 100, line_dash="dot", line_color="#1565C0")
    fig.update_layout(height=max(300, 60 + 34 * len(v)),
                      margin=dict(l=10, r=50, t=30, b=10),
                      xaxis_title="Variance vs plan (%)",
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(gridcolor="#EEF1F4")
    return fig


def efficiency_scatter(df: pd.DataFrame, objective: str, target: float, sym: str):
    """Pacing against return. Revenue mode plots ROAS, lead-gen plots CPA."""
    is_leads = objective == "Leads"
    y_title = f"CPA MTD ({sym})" if is_leads else "ROAS MTD"

    fig = go.Figure(go.Scatter(
        x=(df["Variance"] * 100).fillna(0),
        y=df["Efficiency"],
        mode="markers+text",
        text=df["Campaign"].astype(str),
        textposition="top center",
        marker=dict(size=bubble_sizes(df["Projected Spend"]),
                    color=[STATUS_COLORS.get(s, "#9AA5B1") for s in df["Status"]],
                    opacity=0.75, line=dict(width=1, color="white")),
        hovertemplate="%{text}<br>Variance %{x:.0f}%<br>"
                      + ("CPA " + sym + "%{y:,.2f}" if is_leads else "ROAS %{y:.2f}x")
                      + "<extra></extra>",
    ))
    if np.isfinite(target) and target > 0:
        fig.add_hline(y=target, line_dash="dash", line_color="#2E7D32",
                      annotation_text=(f"Target {money(target, sym, 2)}" if is_leads
                                       else f"Target {target:.1f}x"))
    fig.add_vline(x=0, line_dash="dot", line_color="#9AA5B1")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Variance vs plan (%)", yaxis_title=y_title,
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(gridcolor="#EEF1F4")
    fig.update_yaxes(gridcolor="#EEF1F4")
    return fig


# ---------------------------------------------------------------------------
# Google Ads CSV ingestion
# ---------------------------------------------------------------------------

def _read_ads_csv(raw: bytes) -> pd.DataFrame:
    """Exports vary by encoding, delimiter and preamble depth."""
    combos = [("utf-8", ","), ("utf-16", "\t"), ("utf-16", ","),
              ("utf-8-sig", ","), ("latin-1", ","), ("utf-8", "\t")]
    for enc, sep in combos:
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        lines = text.splitlines()
        for skip in range(0, min(8, len(lines))):
            if "campaign" not in lines[skip].lower():
                continue
            try:
                cand = pd.read_csv(io.StringIO(text), skiprows=skip, sep=sep,
                                   thousands=",", keep_default_na=False)
            except Exception:
                continue
            cols = [str(c).strip() for c in cand.columns]
            if "Campaign" in cols and len(cols) > 3:
                cand.columns = cols
                return cand
    raise ValueError("Could not read this file. Export the Campaigns table from "
                     "Google Ads as CSV without editing it, and make sure Cost "
                     "is one of the columns.")


def parse_google_ads_export(uploaded) -> pd.DataFrame:
    df = _read_ads_csv(uploaded.getvalue())

    def find(*, exact=None, contains=None):
        for c in df.columns:
            low = str(c).lower().strip()
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

    value_col = (find(contains=["conv. value", "conv. time"])
                 or find(exact="conv. value")
                 or find(contains=["conv. value"])
                 or find(contains=["all conv. value"]))
    conv_col = (find(contains=["conversions", "conv. time"])
                or find(exact="conversions")
                or find(exact="all conv."))
    budget_col = find(exact="budget") or find(contains=["daily budget"])
    status_col = find(exact="campaign status") or find(exact="campaign state")
    cur_col = find(exact="currency code")
    type_col = find(exact="budget type")

    names = df["Campaign"].astype(str).str.strip()
    df = df[(~names.str.startswith(("Total", "--", "-"))) & (names != "")].copy()

    period_col = next((c for c in df.columns
                       if str(c).strip() in ("Month", "Day", "Week", "Quarter", "Year")),
                      None)
    multi_period = bool(period_col) and df[period_col].astype(str).str.strip().nunique() > 1

    dropped = 0
    if status_col:
        vals = df[status_col].astype(str).str.strip().str.lower()
        keep = ~vals.isin(["removed", "paused"])
        dropped = int((~keep).sum())
        if keep.any():
            df = df[keep]

    def num(sr):
        return pd.to_numeric(sr.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
                             .replace("", "0"), errors="coerce").fillna(0.0)

    agg = {"Actual Spend MTD": (cost_col, "sum")}
    df[cost_col] = num(df[cost_col])
    for src, dest in ((value_col, "Conv. Value MTD"), (conv_col, "Conversions MTD")):
        if src:
            df[src] = num(df[src])
            agg[dest] = (src, "sum")
    if budget_col:
        df[budget_col] = num(df[budget_col])
        # Budget repeats across a monthly breakdown; take the latest value.
        agg["Daily Budget"] = (budget_col, "max")

    g = df.groupby("Campaign", as_index=False).agg(**agg)
    for col in ("Conv. Value MTD", "Conversions MTD", "Daily Budget"):
        if col not in g.columns:
            g[col] = 0.0

    currencies = []
    if cur_col:
        currencies = sorted({str(v).strip() for v in df[cur_col]
                             if str(v).strip() not in ("", "--", "-")})
    non_daily = []
    if type_col:
        non_daily = sorted({str(v).strip() for v in df[type_col]
                            if str(v).strip().lower() not in ("daily", "", "--", "-")})

    g.attrs.update(multi_period=multi_period, budget_found=bool(budget_col),
                   value_found=bool(value_col), conv_found=bool(conv_col),
                   currencies=currencies, dropped=dropped, non_daily=non_daily)
    return g


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Google Ads Pacing Monitor", page_icon="\U0001F4C8",
                   layout="wide")
st.markdown(f"<style>.block-container{{padding-top:2rem}}"
            f"h1{{color:{INK};font-size:1.85rem!important}}</style>",
            unsafe_allow_html=True)

for key, default in (("campaigns", pd.DataFrame(columns=EDITOR_COLUMNS)),
                     ("currency", "USD")):
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Account")
    account_name = st.text_input("Account or business unit", value="",
                                 placeholder="e.g. Industrial Division")
    currency = st.selectbox("Currency", list(CURRENCY_SYMBOLS),
                            index=list(CURRENCY_SYMBOLS).index(st.session_state.currency)
                            if st.session_state.currency in CURRENCY_SYMBOLS else 0)
    sym = CURRENCY_SYMBOLS.get(currency, "$")

    st.divider()
    st.header("Period")
    today = dt.date.today()
    period = st.date_input("Month to monitor", value=today.replace(day=1))
    year, month = period.year, period.month
    cal_days = calendar.monthrange(year, month)[1]
    current_month = (today.year, today.month) == (year, month)

    if st.checkbox("Use today's date", value=True):
        days_elapsed = today.day if current_month else cal_days
        if not current_month:
            st.caption("Past month selected \u2014 treated as complete.")
    else:
        days_elapsed = st.slider("Calendar days elapsed", 1, cal_days,
                                 min(today.day, cal_days))
    st.metric("Days elapsed", f"{days_elapsed} / {cal_days}")

    threshold = st.slider("Attention threshold (\u00b1 vs plan)", 0.05, 0.30, 0.10, 0.01)

    st.divider()
    st.header("Load actuals")
    upload = st.file_uploader("Google Ads campaign report (.csv)", type=["csv"])
    st.caption("Campaigns \u2192 Download \u2192 CSV. Include Budget, Cost, "
               "Conversions and Conv. value. Set the date range to this month only.")

st.title(f"Google Ads Pacing Monitor{f' \u2014 {account_name}' if account_name else ''}")

# --- ingestion ---
if upload is not None:
    try:
        parsed = parse_google_ads_export(upload)
        base = st.session_state.campaigns
        prior = base.set_index("Campaign") if "Campaign" in base.columns and len(base) else None

        rows = []
        for _, r in parsed.iterrows():
            name = r["Campaign"]
            was = prior.loc[name] if prior is not None and name in prior.index else None
            budget = float(r.get("Daily Budget") or 0.0)
            if budget <= 0 and was is not None:
                budget = float(pd.to_numeric(was.get("Daily Budget"), errors="coerce") or 0)
            value = float(r.get("Conv. Value MTD") or 0.0)
            rows.append({
                "Business Unit": (was.get("Business Unit") if was is not None
                                  else (account_name or "Unassigned")),
                "Campaign": name,
                "Objective": (was.get("Objective") if was is not None
                              else ("Revenue" if value > 0 else "Leads")),
                "Daily Budget": budget,
                "Basis": was.get("Basis") if was is not None else "7-day",
                "Actual Spend MTD": float(r.get("Actual Spend MTD") or 0.0),
                "Conversions MTD": float(r.get("Conversions MTD") or 0.0),
                "Conv. Value MTD": value,
            })
        st.session_state.campaigns = pd.DataFrame(rows, columns=EDITOR_COLUMNS)

        a = parsed.attrs
        if len(a.get("currencies", [])) > 1:
            st.sidebar.error("This export mixes currencies ("
                             + ", ".join(a["currencies"])
                             + "). Cross-currency totals are meaningless \u2014 "
                               "export one account at a time.")
        elif a.get("currencies"):
            st.session_state.currency = a["currencies"][0]
        if a.get("multi_period"):
            st.sidebar.warning("This export spans more than one period, so these "
                               "figures are a multi-month sum, not month-to-date. "
                               "Re-export with the range set to this month.")
        if not a.get("budget_found"):
            st.sidebar.warning("No Budget column found \u2014 enter daily budgets by hand.")
        if not a.get("value_found") and not a.get("conv_found"):
            st.sidebar.warning("No conversions or conversion value found. Spend "
                               "pacing works; the efficiency panel will not.")
        if a.get("non_daily"):
            st.sidebar.warning("Non-daily budget types present ("
                               + ", ".join(a["non_daily"])
                               + "). This model assumes daily budgets.")
        if a.get("dropped"):
            st.sidebar.caption(f"Excluded {a['dropped']} paused/removed rows.")
        st.sidebar.success(f"Loaded {len(rows)} campaigns. Confirm budgets below.")
    except Exception as exc:
        st.sidebar.error(str(exc))

# --- empty state ---
if st.session_state.campaigns.empty:
    st.info("Upload a Google Ads campaign export in the sidebar, or add campaigns "
            "manually below. Objective is set per campaign: **Revenue** compares "
            "spend against ROAS, **Leads** against CPA.")
    if st.button("Load sample rows"):
        st.session_state.campaigns = SAMPLE_ROWS.copy()
        st.rerun()

st.subheader("Campaign inputs")
st.caption("Daily budget and basis come from Google Ads settings. Spend, conversions "
           "and value auto-fill from the upload.")

edited = st.data_editor(
    st.session_state.campaigns,
    num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={
        "Business Unit": st.column_config.TextColumn(width="small"),
        "Campaign": st.column_config.TextColumn(width="medium"),
        "Objective": st.column_config.SelectboxColumn(
            options=OBJECTIVES, width="small",
            help="Revenue = ecommerce, judged on ROAS. "
                 "Leads = lead gen, judged on CPA."),
        "Daily Budget": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
        "Basis": st.column_config.SelectboxColumn(
            options=["7-day", "Weekday"], width="small",
            help="7-day = always on. Weekday = an ad schedule restricts it to Mon-Fri."),
        "Actual Spend MTD": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
        "Conversions MTD": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
        "Conv. Value MTD": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
    },
    key="editor",
)
st.session_state.campaigns = edited

work = edited.copy()
if "Campaign" in work.columns:
    work = work[~work["Campaign"].astype(str).str.strip().isin(("", "None", "nan"))]
if work.empty:
    st.stop()

paced = compute_pacing(work, year, month, days_elapsed, threshold)

# --- BU filter ---
units = sorted(paced["Business Unit"].astype(str).unique())
if len(units) > 1:
    chosen = st.multiselect("Business units", units, default=units)
    paced = paced[paced["Business Unit"].astype(str).isin(chosen)]
    if paced.empty:
        st.warning("No campaigns in the selected business units.")
        st.stop()

st.divider()

# --- headline metrics ---
budgeted = paced[paced["Daily Budget"] > 0]
planned = budgeted["Planned Monthly"].sum()
actual = budgeted["Actual Spend MTD"].sum()
projected = budgeted["Projected Spend"].sum(skipna=True)
variance = (projected - planned) / planned if planned else np.nan

rev = budgeted[budgeted["Objective"].eq("Revenue")]
lead = budgeted[budgeted["Objective"].eq("Leads")]

cols = st.columns(3 + (1 if not rev.empty else 0) + (1 if not lead.empty else 0))
cols[0].metric("Spend MTD", money(actual, sym),
               f"{actual / planned:.0%} of plan" if planned else None)
cols[1].metric("Projected month-end", money(projected, sym),
               f"{variance:+.0%} vs plan" if pd.notna(variance) else None)
cols[2].metric("Planned monthly", money(planned, sym))
i = 3
if not rev.empty:
    r_spend = rev["Actual Spend MTD"].sum()
    roas = rev["Conv. Value MTD"].sum() / r_spend if r_spend else np.nan
    proj_val = rev["Projected Value"].sum(skipna=True)
    cols[i].metric("ROAS MTD", f"{roas:.2f}x" if pd.notna(roas) else "\u2014",
                   f"{money(proj_val, sym)} projected revenue")
    i += 1
if not lead.empty:
    l_spend = lead["Actual Spend MTD"].sum()
    l_conv = lead["Conversions MTD"].sum()
    cpa = l_spend / l_conv if l_conv else np.nan
    proj_leads = lead["Projected Conversions"].sum(skipna=True)
    cols[i].metric("CPA MTD", money(cpa, sym, 2) if pd.notna(cpa) else "\u2014",
                   f"{proj_leads:,.0f} leads projected")

unbudgeted = paced[paced["Daily Budget"] <= 0]
if not unbudgeted.empty:
    st.error("**No daily budget set for: "
             + ", ".join(unbudgeted["Campaign"].astype(str))
             + f"** \u2014 {money(unbudgeted['Actual Spend MTD'].sum(), sym)} of spend "
               "is excluded from every plan and projection below.")

if pd.notna(variance):
    if variance > threshold:
        st.error(f"Pacing **{variance:+.0%} above plan**. Projected overspend: "
                 f"{money(projected - planned, sym)}.")
    elif variance < -threshold:
        st.warning(f"Pacing **{variance:+.0%} below plan**. "
                   f"{money(planned - projected, sym)} will go unspent.")
    else:
        st.success(f"On plan (within {threshold:.0%}).")

# --- BU rollup ---
if len(units) > 1:
    roll = paced.groupby("Business Unit").agg(
        Campaigns=("Campaign", "count"),
        Planned=("Planned Monthly", "sum"),
        Spend=("Actual Spend MTD", "sum"),
        Projected=("Projected Spend", "sum"),
        Conversions=("Conversions MTD", "sum"),
        Value=("Conv. Value MTD", "sum"),
    ).reset_index()
    roll["Variance"] = (roll["Projected"] - roll["Planned"]) / roll["Planned"].replace(0, np.nan)
    roll["ROAS"] = roll["Value"] / roll["Spend"].replace(0, np.nan)
    roll["CPA"] = roll["Spend"] / roll["Conversions"].replace(0, np.nan)
    st.subheader("By business unit")
    st.dataframe(roll.style.format({
        "Planned": f"{sym}{{:,.0f}}", "Spend": f"{sym}{{:,.0f}}",
        "Projected": f"{sym}{{:,.0f}}", "Value": f"{sym}{{:,.0f}}",
        "Conversions": "{:,.1f}", "Variance": "{:+.1%}",
        "ROAS": "{:.2f}x", "CPA": f"{sym}{{:,.2f}}",
    }, na_rep="\u2014"), use_container_width=True, hide_index=True)

# --- detail ---
st.subheader("Campaign detail")
detail = paced[["Business Unit", "Campaign", "Objective", "Daily Budget", "Basis",
                "Planned Monthly", "Actual Spend MTD", "Projected Spend",
                "Variance", "ROAS MTD", "CPA MTD", "Status"]]

st.dataframe(
    detail.style.apply(
        lambda r: [f"color:{STATUS_COLORS.get(r['Status'], INK)};font-weight:600"
                   if c == "Status" else "" for c in r.index], axis=1
    ).format({
        "Daily Budget": f"{sym}{{:,.2f}}", "Planned Monthly": f"{sym}{{:,.0f}}",
        "Actual Spend MTD": f"{sym}{{:,.2f}}", "Projected Spend": f"{sym}{{:,.0f}}",
        "Variance": "{:+.1%}", "ROAS MTD": "{:.2f}x", "CPA MTD": f"{sym}{{:,.2f}}",
    }, na_rep="\u2014"),
    use_container_width=True, hide_index=True,
)

left, right = st.columns([3, 2])
with left:
    st.markdown("**Delivery curve**")
    st.plotly_chart(delivery_curve(paced, year, month, days_elapsed, sym),
                    use_container_width=True)
with right:
    st.markdown("**Variance vs plan**")
    st.plotly_chart(variance_bars(paced, threshold), use_container_width=True)

# --- efficiency ---
st.divider()
st.subheader("Pacing vs return")
st.caption("Hitting a budget target is only good news if the money is working. "
           "Bubble size = projected month-end spend.")

tabs, frames = [], []
rev_plot = paced[paced["Objective"].eq("Revenue") & paced["ROAS MTD"].notna()
                 & (paced["Actual Spend MTD"] > 0)]
lead_plot = paced[paced["Objective"].eq("Leads") & paced["CPA MTD"].notna()
                  & (paced["Actual Spend MTD"] > 0)]
if not rev_plot.empty:
    tabs.append("Revenue (ROAS)"); frames.append(("Revenue", rev_plot))
if not lead_plot.empty:
    tabs.append("Lead gen (CPA)"); frames.append(("Leads", lead_plot))

if not frames:
    st.info("No campaigns with both spend and conversion data yet.")
else:
    for tab, (objective, frame) in zip(st.tabs(tabs), frames):
        with tab:
            if objective == "Revenue":
                target = st.number_input("Target ROAS", 0.5, 50.0, 4.0, 0.5,
                                         key="t_roas")
                guide = ("Bottom-right is the expensive quadrant: spending ahead of "
                         "plan below target ROAS. Top-left is money left on the "
                         "table \u2014 earning well but underspending.")
            else:
                default_cpa = float(np.nanmedian(frame["CPA MTD"])) or 50.0
                target = st.number_input(f"Target CPA ({sym})", 0.0, 100000.0,
                                         round(default_cpa, 2), 1.0, key="t_cpa")
                guide = ("Lower is better here, so the quadrants flip: **top-right** "
                         "is the expensive one \u2014 spending ahead of plan at a CPA "
                         "above target. Bottom-left is an underfunded winner: cheap "
                         "leads you are not buying enough of.")
            st.plotly_chart(efficiency_scatter(frame, objective, target, sym),
                            use_container_width=True)
            st.markdown(guide)

# --- export ---
st.divider()
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xl:
    paced.to_excel(xl, sheet_name="Pacing", index=False)
st.download_button(
    "Download this view as Excel", buf.getvalue(),
    file_name=f"{slugify(account_name or 'account')}_pacing_{year}-{month:02d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("How the numbers are calculated"):
    st.markdown(f"""
**Plan**

- **Planned Monthly** = daily budget \u00d7 scheduled days. A **7-day** campaign is
  planned against Google's billable month ({BILLABLE_MULTIPLIER} days); a **Weekday**
  campaign against the **{weekdays_in_month(year, month)}** weekdays in this month.
- **Monthly Cap** = daily budget \u00d7 {BILLABLE_MULTIPLIER} \u2014 Google's billing
  ceiling. Projections are capped there because you cannot be charged above it.

**Projection**

- Spend, conversions and value all project as MTD \u00f7 elapsed days \u00d7 scheduled
  days, using **{days_elapsed}** calendar or
  **{weekdays_elapsed(year, month, days_elapsed)}** weekday days elapsed to match
  each campaign's basis.
- **Variance** = (Projected \u2212 Planned) \u00f7 Planned, flagged at
  \u00b1{threshold:.0%}.
- **At cap** means the raw run-rate exceeds the billing ceiling \u2014 the campaign is
  demand-limited, not budget-limited. Raising the budget is what moves it.

**Two things worth knowing**

1. For an always-on campaign the plan *is* the cap, so it can only ever pace
   **under** plan. Genuine overspend risk exists only where an ad schedule
   restricts delivery. A monitor that plans on weekdays but projects on calendar
   days manufactures a permanent \u2248{BILLABLE_MULTIPLIER / 22 - 1:.0%} false
   overspend flag on every always-on campaign.
2. Objective is set per campaign. Lead-gen campaigns are judged on CPA, where
   **lower is better** \u2014 so a lead-gen campaign pacing under plan at a low CPA
   is an opportunity, not a problem.
""")
