# Creates a SYNTHETIC dataset with the same structure as the real input file.
# Values are randomly generated and do NOT represent any real household.
import sys
import numpy as np
import pandas as pd

n_houses = int(sys.argv[1]) if len(sys.argv) > 1 else 40
out = sys.argv[2] if len(sys.argv) > 2 else "sample_data_synthetic.xlsx"
rng = np.random.default_rng(123)
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

props = pd.DataFrame({
    "Property_Address": [f"SYNTHETIC HOUSE {i + 1:03d}" for i in range(n_houses)],
    "Total Area": rng.uniform(300, 1500, n_houses).round(0),
    "Roof Area": rng.uniform(100, 350, n_houses).round(0),
    "Grass Area": np.where(rng.random(n_houses) < 0.1, 0, rng.uniform(20, 250, n_houses)).round(0),
    "Pool": rng.choice(["Yes", "No"], n_houses, p=[0.4, 0.6]),
    "Age": rng.uniform(30, 65, n_houses).round(0),
    "Weekly Income": rng.uniform(1700, 2600, n_houses).round(0),
    "lat": -26.65 + rng.normal(0, 0.05, n_houses),
    "lon": 153.07 + rng.normal(0, 0.05, n_houses),
})

doy = dates.dayofyear.to_numpy()
season = np.cos(2 * np.pi * (doy - 15) / 365)          # summer peak (southern hemisphere)
clim = pd.DataFrame({
    "Date": dates.strftime("%d/%m/%Y"),
    "Rainfall": np.where(rng.random(len(dates)) < 0.35, rng.gamma(1.2, 8, len(dates)), 0).round(1),
    "Soil moisture": (45 + 3 * season + rng.normal(0, 1.5, len(dates))).round(1),
    "Min Temperature": (16 + 5 * season + rng.normal(0, 2, len(dates))).round(1),
    "Max Temperature": (25 + 5 * season + rng.normal(0, 2.5, len(dates))).round(1),
    "Evaporation": np.clip(5 + 2.5 * season + rng.normal(0, 1.0, len(dates)), 0.5, None).round(1),
})

# daily evaporation-driven pool top-up (L/day) for a representative 32 m2 pool
pool = pd.DataFrame({"Date": clim["Date"], "pool_topup": (clim["Evaporation"] * 32.0).round(0)})

use = {}
for i in range(n_houses):
    g = props.loc[i, "Grass Area"]
    base = (2 + 2 * season) * g * rng.uniform(0.6, 1.8)
    water_day = rng.random(len(dates)) < rng.uniform(0.2, 0.6)
    val = np.where(water_day, base * rng.lognormal(0, 0.4, len(dates)), 0)
    val = val * np.where(clim["Rainfall"] > 5, 0.2, 1.0)
    if props.loc[i, "Pool"] == "Yes":
        val = val + np.where(rng.random(len(dates)) < 0.15, rng.uniform(100, 400, len(dates)), 0)
    use[i] = val.round(0)
usage = pd.DataFrame(use).T
usage.columns = dates
usage.insert(0, "Property_Address", props["Property_Address"])

with pd.ExcelWriter(out, engine="openpyxl") as w:
    props.to_excel(w, sheet_name="property detail", index=False)
    usage.to_excel(w, sheet_name="daily outdoor use", index=False)
    clim.to_excel(w, sheet_name="Climic data", index=False)
    pool.to_excel(w, sheet_name="Weekly Pool Topup", index=False)
print(f"Wrote {out} ({n_houses} synthetic households)")
