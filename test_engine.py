"""Headless smoke test of the engine against the sample exports."""
from pathlib import Path
import pandas as pd
from engine import load_report, aggregate, verdict, pacing_table, ngrams

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
assert len(mt) == 3

st_agg = aggregate(files["search_terms"], ["search_term"])
print("\nsearch terms agg:", st_agg.shape)

# impression share sentinels parsed?
print("\nlost_is_budget sample:", files["campaigns"]["lost_is_budget"].dropna().unique()[:6])
assert files["campaigns"]["lost_is_budget"].max() <= 1.0

print("\nALL CHECKS PASSED")
