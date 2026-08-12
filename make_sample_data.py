"""
Generates sample CSVs that mimic real Google Ads UI exports — including the
junk preamble rows, currency symbols, '--' nulls and '< 10%' impression-share
sentinels — so you can verify the app runs before loading live account data.

    python make_sample_data.py
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(7)
np.random.seed(7)

OUT = Path(__file__).parent / "sample_data"
OUT.mkdir(exist_ok=True)

CATS = ["Cat6", "Cat6a", "Cat5e", "Cat8", "HDMI 2.1", "USB-C", "Fiber LC-LC",
        "RG-58 Coax", "RG-142 Coax", "D-Sub", "QSFP28 DAC", "SFP+ AOC"]
LENS = ["3ft", "10ft", "25ft", "50ft", "100ft", "500ft Spool", "1000ft Spool", "Custom"]
TYPES = ["Search", "Performance Max", "Shopping", "Display"]


def preamble(name, rows, cols, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        f.write("Aug 1, 2026 - Aug 31, 2026\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join('"' + str(x) + '"' for x in r) + "\n")
        total = ["Total: all"] + ["" for _ in cols[1:]]
        f.write(",".join(total) + "\n")


campaigns = []
rows = []
for i in range(320):
    cat = random.choice(CATS)
    ln = random.choice(LENS)
    ctype = random.choices(TYPES, weights=[6, 2, 2, 1])[0]
    name = f"{ctype[:3].upper()} | {cat} | {ln}"
    campaigns.append(name)
    heavy = i < 12
    budget = round(random.choice([5, 10, 15, 25, 40, 75, 150] + ([300, 500] if heavy else [])), 2)
    impr = int(np.random.lognormal(7.5 if heavy else 5.2, 1.1))
    clicks = int(impr * random.uniform(0.005, 0.09))
    cpc = random.uniform(0.6, 4.5)
    cost = round(clicks * cpc, 2)
    cvr = random.uniform(0, 0.07)
    conv = round(clicks * cvr, 2)
    val = round(conv * random.uniform(45, 400), 2)
    lost_b = random.choice(["--", "< 10%", "12.44%", "31.09%", "5.20%", "> 90%"])
    rows.append([name, random.choice(["Enabled", "Enabled", "Paused"]), ctype,
                 random.choice(["Maximize conversion value", "Target CPA", "Manual CPC"]),
                 f"${budget:,.2f}", f"${cost:,.2f}", f"{clicks:,}", f"{impr:,}",
                 f"{conv:,.2f}", f"${val:,.2f}",
                 random.choice(["--", "43.10%", "67.55%", "< 10%"]), lost_b,
                 random.choice(["--", "22.10%", "51.02%"])])

preamble("Campaign report", rows,
         ["Campaign", "Campaign state", "Campaign type", "Bid strategy type", "Budget", "Cost",
          "Clicks", "Impr.", "Conversions", "Conv. value", "Search impr. share",
          "Search lost IS (budget)", "Search lost IS (rank)"],
         OUT / "campaigns.csv")

# Ad groups
ag_rows = []
for c in campaigns:
    for j in range(random.randint(1, 4)):
        clicks = int(np.random.lognormal(3.4, 1.2))
        cost = round(clicks * random.uniform(0.6, 4.0), 2)
        conv = round(clicks * random.uniform(0, 0.06), 2)
        ag_rows.append([c, f"AG {j+1} - {random.choice(LENS)}", "Enabled",
                        f"${cost:,.2f}", f"{clicks:,}", f"{int(clicks*random.uniform(8,40)):,}",
                        f"{conv:,.2f}", f"${conv*random.uniform(50,350):,.2f}"])
preamble("Ad group report", ag_rows,
         ["Campaign", "Ad group", "Ad group state", "Cost", "Clicks", "Impr.",
          "Conversions", "Conv. value"], OUT / "ad_groups.csv")

# Keywords
kw_rows = []
for c in random.sample(campaigns, 160):
    for _ in range(random.randint(1, 6)):
        cat = random.choice(CATS).lower()
        kw = f"{random.choice(['buy ', 'bulk ', '', 'cheap ', ''])}{cat} cable {random.choice(LENS).lower()}"
        clicks = int(np.random.lognormal(2.6, 1.3))
        cost = round(clicks * random.uniform(0.5, 5.0), 2)
        conv = round(clicks * random.uniform(0, 0.05), 2)
        kw_rows.append([c, f"AG 1 - {random.choice(LENS)}", kw,
                        random.choice(["Exact match", "Phrase match", "Broad match"]),
                        f"${cost:,.2f}", f"{clicks:,}", f"{int(clicks*random.uniform(9,60)):,}",
                        f"{conv:,.2f}", f"${conv*random.uniform(50,350):,.2f}",
                        random.choice(["--", "3", "5", "7", "8", "10"])])
preamble("Keyword report", kw_rows,
         ["Campaign", "Ad group", "Keyword", "Match type", "Cost", "Clicks", "Impr.",
          "Conversions", "Conv. value", "Quality score"], OUT / "keywords.csv")

# Search terms
st_rows = []
JUNK = ["free", "diagram", "how to make", "pinout", "used", "wiring", "schematic", "repair"]
for _ in range(4000):
    cat = random.choice(CATS).lower()
    bits = [cat, "cable"]
    if random.random() < 0.35:
        bits.insert(0, random.choice(JUNK))
    if random.random() < 0.5:
        bits.append(random.choice(LENS).lower())
    term = " ".join(bits)
    clicks = max(1, int(np.random.lognormal(1.2, 1.0)))
    cost = round(clicks * random.uniform(0.4, 4.0), 2)
    junky = any(j in term for j in JUNK)
    conv = 0 if junky or random.random() < 0.8 else round(clicks * random.uniform(0.02, 0.12), 2)
    st_rows.append([term, random.choice(campaigns), random.choice(["Exact match", "Phrase match", "Broad match"]),
                    random.choice(["None", "Added", "Excluded"]),
                    f"${cost:,.2f}", f"{clicks:,}", f"{int(clicks*random.uniform(10,80)):,}",
                    f"{conv:,.2f}", f"${conv*random.uniform(50,350):,.2f}"])
preamble("Search terms report", st_rows,
         ["Search term", "Campaign", "Match type", "Added/Excluded", "Cost", "Clicks",
          "Impr.", "Conversions", "Conv. value"], OUT / "search_terms.csv")

# Ads
ad_rows = []
for c in random.sample(campaigns, 120):
    for k in range(random.randint(1, 3)):
        clicks = int(np.random.lognormal(2.8, 1.1))
        cost = round(clicks * random.uniform(0.5, 4.0), 2)
        conv = round(clicks * random.uniform(0, 0.06), 2)
        ad_rows.append([c, "AG 1 - 10ft", f"RSA {k+1}", "Responsive search ad",
                        "https://cablesondemand.com/", f"${cost:,.2f}", f"{clicks:,}",
                        f"{int(clicks*random.uniform(10,60)):,}", f"{conv:,.2f}",
                        f"${conv*random.uniform(50,350):,.2f}"])
preamble("Ad report", ad_rows,
         ["Campaign", "Ad group", "Ad", "Ad type", "Final URL", "Cost", "Clicks",
          "Impr.", "Conversions", "Conv. value"], OUT / "ads.csv")

print("Wrote sample files to", OUT)
for p in sorted(OUT.glob("*.csv")):
    print(" -", p.name, f"{p.stat().st_size/1024:,.0f} KB")
