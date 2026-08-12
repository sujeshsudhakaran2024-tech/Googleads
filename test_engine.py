"""Headless smoke test of the engine against the sample exports."""
from pathlib import Path
import pandas as pd
from engine import load_report, aggregate, verdict, pacing_table, ngrams, join_cols, safe_str

D = Path("sample_data")
files = {p.stem: load_report(p.read_bytes(), p.name) for p in D.glob("*.csv")}
for k, df in files.items():
    print(f"{k:14s} rows={len(df):6,d} cols={list(df.columns)[:6]}")

camp = aggregate(files["campaigns"], ["campaign", "campaign_type", "bid_strategy"])
print("\ncampaign agg:", camp.shape, "cost=", round(camp.cost.sum(), 2), "conv=", round(camp.conversions.sum(),2), "val=", round(camp.conv_value.sum(),2))
assert camp.cost.sum() > 0 and camp.conv_value.sum() > 0
assert not camp["campaign"].astype(str).str.lower().str.startswith("total").any(), "total row leaked"

cvr = camp.conversions.sum()/camp.clicks.sum()
v = verdict(camp, cvr, 4.0, 95.0, 3.0, True)
print("\nverdict counts:\n", v.verdict.value_counts())
print("dollars at stake:", round(v.dollars_at_stake.sum(), 2))
assert v.dollars_at_stake.sum() > 0

p = pacing_table(camp, 31, 31)
print("\npacing:\n", p.pacing_status.value_counts())
assert p.budget.notna().any(), "budget column lost"
assert p.pace_vs_budget.notna().any()

g = ngrams(files["search_terms"], "search_term", n=2, min_cost=10)
print("\ntop waste bigrams:\n", g[g.conversions==0].head(8)[["ngram","cost","clicks","conversions"]])
assert not g.empty

kw = aggregate(files["keywords"], ["campaign","ad_group","keyword","match_type"])
mt = aggregate(files["keywords"], ["match_type"])
print("\nmatch types:\n", mt[["match_type","cost","conversions","roas"]])
assert len(mt) >= 3

st_agg = aggregate(files["search_terms"], ["search_term"])
print("\nsearch terms agg:", st_agg.shape)

# impression share sentinels parsed?
print("\nlost_is_budget sample:", files["campaigns"]["lost_is_budget"].dropna().unique()[:6])
assert files["campaigns"]["lost_is_budget"].max() <= 1.0


# --- regression: pandas 3 keeps NA through astype(str), which broke row labels ---
dirty = pd.DataFrame({"campaign": ["A", None, "C"], "ad_group": [None, "y", "z"]})
lbl = join_cols(dirty, ["campaign", "ad_group"])
print("\njoin_cols on NaN keys:", lbl.tolist())
assert lbl.tolist() == ["A › ", " › y", "C › z"]
assert join_cols(dirty, ["campaign"]).tolist() == ["A", "", "C"]
assert join_cols(dirty, ["missing_col"]).tolist() == ["", "", ""]
assert safe_str(pd.Series([1.0, None])).tolist() == ["1.0", ""]

# blank label cells must survive the whole pipeline
kw2 = aggregate(files["keywords"], ["campaign", "ad_group", "keyword", "match_type"])
v2 = verdict(kw2, cvr, 4.0, 95.0, 3.0, True)
v2["entity"] = join_cols(v2, ["campaign", "ad_group", "keyword", "match_type"])
assert v2["entity"].map(type).eq(str).all()
print("blank-cell rows survived:", int(kw2["match_type"].isna().sum() + (kw2["match_type"] == "").sum()))

# empty input must not explode
assert verdict(kw2.head(0), cvr, 4.0, 95.0, 3.0, True).empty
print("\nALL CHECKS PASSED")
