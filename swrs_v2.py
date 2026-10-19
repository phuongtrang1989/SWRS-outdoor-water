# =====================================================================
# SMART WATER RECOMMENDATION SYSTEM (SWRS) 
# =====================================================================

import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from sklearn.decomposition import TruncatedSVD
from scipy.stats import wilcoxon
from scipy.optimize import linear_sum_assignment


def main():

    t0 = time.time()

    # =====================================================================
    # 0. SETTINGS
    # =====================================================================
    DATA_PATH = str(Path(__file__).resolve().parent / "data" / "sample_data_synthetic.xlsx")
    if len(sys.argv) > 1:                     
        DATA_PATH = sys.argv[1]

    SHEET_PROPS = "property detail"
    SHEET_USAGE = "daily outdoor use"        
    SHEET_CLIM = "Climic data"

    ANCHOR = pd.Timestamp("2023-01-01")       # Sunday -> week 1 starts here
    TEST_START_REQUESTED = pd.Timestamp("2024-06-01")
    LAG_SHORT = 30                            # days, behavioural window
    LAG_LONG = 180                            # days, seasonal window
    MIN_DAYS_PER_WEEK = 5                     # min. valid days for a household-week to be evaluated

    SEEDS = [42, 1, 2, 3, 4]                  # first seed = main reported run
    RUN_FULL_FACTORIAL = True                 # False -> only the 4 configurations of Table 2
    COHERENCE_K = list(range(3, 11))          # [] to skip coherence analysis
    COHERENCE_SAMPLE = 20000                  # documents used to compute C_v
    N_BOOT = 1000                             # household-level bootstrap replicates
    ALERT_THRESHOLDS = [1.2, 1.3, 1.5]
    CASE_WEEK_SEQ = 98                        # illustrative week (10-16 Nov 2024)
    ANONYMISE_IDS = True                      # True -> HH001, HH002 ... in outputs

    CFG = {
        # ===== Property bins (from CART) =====
        "lot_bins": [-np.inf, 720.0, 770.0, 1240.0, np.inf],
        "lot_labels": ["low", "mid", "high", "very_high"],
        "roof_bins": [-np.inf, 140.0, 260.0, 300.0, np.inf],
        "roof_labels": ["low", "mid", "high", "very_high"],
        "grass_bins": [-np.inf, 0.0, 60.0, 90.0, 170.0, np.inf],
        "grass_labels": ["none", "small", "medium", "large", "xlarge"],
        "income_bins": [-np.inf, 1950.0, 2350.0, 2400.0, np.inf],
        "income_labels": ["low", "mid", "high", "very_high"],
        "age_bins": [-np.inf, 40.0, 55.0, np.inf],
        "age_labels": ["low", "mid", "high"],
        # ===== Usage bins (from CART), L/day =====
        "use_bins": [-np.inf, 1310.0, 6310.0, 15290.0, np.inf],
        "use_labels": ["low", "mid", "high", "very_high"],
        # ===== Climate bins (from CART) =====
        "rain_bins": [-np.inf, 0.0, 8.0, np.inf],
        "rain_labels": ["none", "small", "medium"],
        "tmax_bins": [-np.inf, 19.0, 20.0, 30.0, np.inf],
        "tmax_labels": ["low", "mid", "high", "very_high"],
        "tmin_bins": [-np.inf, 22.0, np.inf],
        "tmin_labels": ["low", "mid"],
        "soil_bins": [-np.inf, 44.0, 46.0, np.inf],
        "soil_labels": ["low", "mid", "high"],
        # ===== Other =====
        "intensity_cut": [2, 5],              # L/m2/day: <2 low; 2-5 med; >5 high
        # ===== LDA =====
        "no_below": 5,
        "no_above": 0.90,
        "num_topics": 6,
        "passes": 10,
    }

    IRR_EFF = 1.0                             # assumed irrigation efficiency
    # ---- Pool top-up (Reviewer 1, comment 6) ----
    # "water_balance": weekly net loss  max(sum(E) - sum(P), 0) x pool area, spread
    #                  evenly over the days of the week (rainfall on the pool surface
    #                  offsets evaporation within the same week)
    # "rule":          previous rule 200 L/day x heat factor x rain factor
    # "sheet":         read daily pool top-up (L/day) from a sheet with columns Date, pool_topup
    # "sheet_net":     the sheet holds evaporation x pool area (no rainfall); evaporation is
    #                  recovered as sheet / POOL_AREA_M2 and the weekly water balance
    #                  max(sum E - sum P, 0) x POOL_AREA_M2 is applied (recommended)
    POOL_METHOD = "sheet_net"
    POOL_SHEET = "Weekly Pool Topup"          
    POOL_AREA_M2 = 32.0                       # representative pool surface area (m2)
    EVAP_COL = "Evaporation"                  # daily evaporation (mm/day) in the climate sheet
    POOL_TOPUP_HOT_NO_RAIN = 200.0            # L/day, only used when POOL_METHOD = "rule"
    HEAT_FACTOR = {"very_high": 1.00, "high": 0.85, "mid": 0.60, "low": 0.40}
    RAIN_FACTOR = {"none": 1.00, "small": 0.50, "medium": 0.00, "heavy": 0.00, "very_heavy": 0.00}
    BASELINE_CLIP_MM = (0.0, 15.0)            # sanity clamp of daily baseline (mm)

    # CF weight in the peer/CF blend as a function of the number of same-day peers
    def cf_weight(n_peers):
        n = np.asarray(n_peers, dtype=float)
        w = np.select([np.isnan(n) | (n <= 2), n <= 4, n <= 9], [0.75, 0.55, 0.35], 0.20)
        return w

    CART_TOKEN_COLS = ["tok_total_area", "tok_roof", "tok_grass", "tok_pool", "tok_income",
                       "tok_age", "tok_tmax", "tok_tmin", "tok_rain", "tok_soil"]
    CART_PREFIXES = ["total_area_", "roof_", "grass_", "pool_", "income_",
                     "age_", "tmax_", "tmin_", "rain_", "soil_"]

    # =====================================================================
    # 1. LOAD + CLEAN DATA
    # =====================================================================
    print("Loading data ...")
    props = pd.read_excel(DATA_PATH, sheet_name=SHEET_PROPS, engine="openpyxl")
    usage_wide = pd.read_excel(DATA_PATH, sheet_name=SHEET_USAGE, engine="openpyxl")
    clim = pd.read_excel(DATA_PATH, sheet_name=SHEET_CLIM, engine="openpyxl")


    def rename_first(df, cands, to):
        for c in cands:
            if c in df.columns and c != to:
                df.rename(columns={c: to}, inplace=True)
                return


    def norm_id_col(df):
        if "Property_Address" in df.columns:
            df["Property_Address"] = (df["Property_Address"].astype(str).str.strip().str.upper()
                                      .str.replace(r"\s+", " ", regex=True))
            df["house_id"] = df["Property_Address"]
        else:
            df["house_id"] = (df.index + 1).astype(str)


    for d in (props, usage_wide):
        norm_id_col(d)

    props.columns = props.columns.str.replace("\u00A0", " ", regex=False).str.strip()
    rename_first(props, ["Total Area", "Total area", "Lot Area", "Site Area", "total area", "total_area_m2"], "Total Area")
    rename_first(props, ["Roof Area", "roof area", "roof_area_m2"], "Roof Area")
    rename_first(props, ["Grass Area", "grass area", "grass_area_m2", "lawn area", "turf area"], "Grass Area")
    rename_first(props, ["Pool", "pool", "pool_yesno"], "Pool")
    rename_first(props, ["Age", "age", "household_age"], "Age")
    rename_first(props, ["Weekly Income", "weekly income", "income", "income_aud_week"], "Weekly Income")
    for c in ["Total Area", "Roof Area", "Grass Area", "Age", "Weekly Income"]:
        if c in props.columns:
            props[c] = pd.to_numeric(props[c], errors="coerce")


    def parse_header_date(col):
        if isinstance(col, pd.Timestamp):
            return col.normalize()
        if isinstance(col, (int, float)) and col > 20000:
            return (pd.to_datetime("1899-12-30") + pd.to_timedelta(int(col), unit="D")).normalize()
        try:
            return pd.to_datetime(str(col), dayfirst=True, errors="raise").normalize()
        except Exception:
            return pd.NaT


    date_map = {c: parse_header_date(c) for c in usage_wide.columns}
    date_cols = [c for c, d in date_map.items() if pd.notna(d)]
    usage_wide.replace(r"^#+$", np.nan, regex=True, inplace=True)
    id_cols = ["house_id"] + (["Property_Address"] if "Property_Address" in usage_wide.columns else [])
    usage_long = usage_wide.melt(id_vars=id_cols, value_vars=date_cols,
                                 var_name="Date_raw", value_name="outdoor_liters")
    usage_long["Date"] = pd.to_datetime(usage_long["Date_raw"].map(date_map))
    usage_long.drop(columns=["Date_raw"], inplace=True)
    usage_long["outdoor_liters"] = pd.to_numeric(usage_long["outdoor_liters"], errors="coerce")
    if "Property_Address" in usage_long.columns:
        usage_long = usage_long.drop(columns=["Property_Address"])

    rename_first(clim, ["Soil moisture", "Soil Moisture", "soil moisture"], "Soil moisture")
    rename_first(clim, ["Rainfall", "Rain fall", "rainfall"], "Rainfall")
    rename_first(clim, ["Min Temperature", "Min temp", "min temp"], "Min Temperature")
    rename_first(clim, ["Max Temperature", "Max temp", "max temp"], "Max Temperature")
    rename_first(clim, ["Evaporation", "evaporation", "Evap", "evap", "Evaporation (mm)",
                        "Pan evaporation", "Potential evaporation", "PET"], EVAP_COL)
    clim["Date"] = pd.to_datetime(clim["Date"], dayfirst=True, errors="coerce")
    for c in ["Soil moisture", "Rainfall", "Min Temperature", "Max Temperature", EVAP_COL]:
        if c in clim.columns:
            clim[c] = pd.to_numeric(clim[c], errors="coerce")
    pool_daily = None
    if POOL_METHOD in ("sheet", "sheet_net"):
        pool_daily = pd.read_excel(DATA_PATH, sheet_name=POOL_SHEET, engine="openpyxl")
        cols = {str(c).strip().lower(): c for c in pool_daily.columns}
        if not {"date", "pool_topup"}.issubset(cols):
            raise ValueError(f"Sheet '{POOL_SHEET}' must have columns 'Date' and 'pool_topup'; found {list(pool_daily.columns)}")
        pool_daily = pool_daily.rename(columns={cols["date"]: "Date", cols["pool_topup"]: "pool_topup"})
        pool_daily["Date"] = pd.to_datetime(pool_daily["Date"], dayfirst=True, errors="coerce").dt.normalize()
        pool_daily["pool_topup"] = pd.to_numeric(pool_daily["pool_topup"].astype(str)
                                                 .str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        pool_daily = pool_daily.dropna(subset=["Date"]).groupby("Date")["pool_topup"].sum()
        print(f"Pool top-up read from sheet '{POOL_SHEET}': {len(pool_daily)} days")
    if POOL_METHOD == "water_balance" and EVAP_COL not in clim.columns:
        raise ValueError(f"POOL_METHOD='water_balance' needs a daily evaporation column named "
                         f"'{EVAP_COL}' (mm/day) in sheet '{SHEET_CLIM}'. Columns found: {list(clim.columns)}")

    usage = (usage_long.merge(props, on="house_id", how="left")
             .merge(clim, on="Date", how="left")
             .sort_values(["house_id", "Date"]).reset_index(drop=True))

    # Week index (Sun-Sat, week 1 starts 2023-01-01) and train/test split
    usage["week_seq"] = ((usage["Date"] - ANCHOR).dt.days // 7) + 1
    usage["week_start"] = ANCHOR + pd.to_timedelta((usage["week_seq"] - 1) * 7, unit="D")
    TEST_START = ANCHOR + pd.Timedelta(days=int(np.ceil((TEST_START_REQUESTED - ANCHOR).days / 7) * 7))
    usage["is_train"] = usage["Date"] < TEST_START
    print(f"Rows: {len(usage):,} | houses: {usage['house_id'].nunique()} | "
          f"train < {TEST_START.date()} <= test")

    # =====================================================================
    # 2. KEYWORDS (static + climate of the day +  behaviour)
    # =====================================================================
    def bin_tokens(series, bins, labels, prefix):
        s = pd.to_numeric(series, errors="coerce")
        cat = pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=True)
        out = (prefix + "_" + cat.astype(str)).astype(object)
        out[s.isna()] = f"{prefix}_unknown"
        return out


    def yn_tokens(series, prefix):
        st = series.astype(str).str.strip().str.lower()
        num = pd.to_numeric(series, errors="coerce")
        yes = st.isin({"1", "y", "yes", "true"}) | (~st.isin({"0", "n", "no", "false"}) & (num > 0))
        no = st.isin({"0", "n", "no", "false"}) | (num.notna() & (num <= 0))
        return pd.Series(np.where(yes, f"{prefix}_yes", np.where(no, f"{prefix}_no", f"{prefix}_unknown")),
                         index=series.index, dtype=object)


    print("Building keywords ...")
    tok = pd.DataFrame(index=usage.index)
    has_area = usage["Total Area"].notna() if "Total Area" in usage else pd.Series(False, index=usage.index)
    if "Total Area" in usage:
        tok["tok_lot"] = bin_tokens(usage["Total Area"], CFG["lot_bins"], CFG["lot_labels"], "lot").where(has_area, None)
        tok["tok_total_area"] = bin_tokens(usage["Total Area"], CFG["lot_bins"], CFG["lot_labels"], "total_area").where(has_area, None)
    if "Roof Area" in usage:
        tok["tok_roof"] = bin_tokens(usage["Roof Area"], CFG["roof_bins"], CFG["roof_labels"], "roof")
    if "Grass Area" in usage:
        tok["tok_grass"] = bin_tokens(usage["Grass Area"], CFG["grass_bins"], CFG["grass_labels"], "grass")
    if "Pool" in usage:
        tok["tok_pool"] = yn_tokens(usage["Pool"], "pool")
    if "Weekly Income" in usage:
        tok["tok_income"] = bin_tokens(usage["Weekly Income"], CFG["income_bins"], CFG["income_labels"], "income")
    if "Age" in usage:
        tok["tok_age"] = bin_tokens(usage["Age"], CFG["age_bins"], CFG["age_labels"], "age")
    if "Rainfall" in usage:
        tok["tok_rain"] = bin_tokens(usage["Rainfall"], CFG["rain_bins"], CFG["rain_labels"], "rain")
    if "Soil moisture" in usage:
        tok["tok_soil"] = bin_tokens(usage["Soil moisture"], CFG["soil_bins"], CFG["soil_labels"], "soil")
    if "Min Temperature" in usage:
        tok["tok_tmin"] = bin_tokens(usage["Min Temperature"], CFG["tmin_bins"], CFG["tmin_labels"], "tmin")
    if "Max Temperature" in usage:
        tok["tok_tmax"] = bin_tokens(usage["Max Temperature"], CFG["tmax_bins"], CFG["tmax_labels"], "tmax")


    def use_token(mean_l):
        if pd.isna(mean_l):
            return "use_unknown"
        lab = pd.cut([mean_l], bins=CFG["use_bins"], labels=CFG["use_labels"], right=True, include_lowest=True)[0]
        return f"use_{lab}"


    def intensity_token(mean_l, grass):
        if pd.isna(mean_l) or pd.isna(grass) or grass <= 0:
            return "intensity_unknown"
        lpm2 = float(mean_l) / max(float(grass), 1.0)
        lo, hi = CFG["intensity_cut"]
        return "intensity_low" if lpm2 < lo else ("intensity_med" if lpm2 <= hi else "intensity_high")


    def freq_token(dates, liters):
        days_used = np.sum(liters > 0)
        span = (dates.max() - dates.min()).astype("timedelta64[D]").astype(int) + 1
        f = days_used / max(span / 7.0, 1e-6)
        return "freq_low" if f <= 1.0 else ("freq_med" if f <= 3.0 else "freq_high")


    def weekend_token(dow, liters):
        wknd = np.nansum(liters[dow >= 5])
        wkdy = np.nansum(liters[dow < 5])
        return "weekend_peak" if wknd > wkdy else None


    def seasonal_token(dates, liters):
        if len(np.unique(dates)) < 30:
            return None
        mid = dates.min() + (dates.max() - dates.min()) / 2
        e, l = liters[dates <= mid], liters[dates > mid]
        early = np.nanmean(e) if np.isfinite(e).any() else np.nan
        late = np.nanmean(l) if np.isfinite(l).any() else np.nan
        if np.isnan(early) or np.isnan(late):
            return None
        return "seasonal_sensitive" if abs(early - late) / max(1.0, (early + late) / 2) > 0.5 else "seasonal_stable"


    # Behavioural tokens are identical for all days of a household-week and use
 
    beh_rows = []
    for hid, g in usage.groupby("house_id", sort=False):
        g = g.sort_values("Date")
        dates = g["Date"].to_numpy()
        liters = g["outdoor_liters"].to_numpy(dtype=float)
        dow = g["Date"].dt.dayofweek.to_numpy()
        grass = g["Grass Area"].iloc[0] if "Grass Area" in g else np.nan
        for ws in np.unique(g["week_start"].to_numpy()):
            hi = np.searchsorted(dates, ws, side="left")
            lo30 = np.searchsorted(dates, ws - np.timedelta64(LAG_SHORT, "D"), side="left")
            lo180 = np.searchsorted(dates, ws - np.timedelta64(LAG_LONG, "D"), side="left")
            rec = {"house_id": hid, "week_start": ws, "tok_use": None, "tok_intensity": None,
                   "tok_freq": None, "tok_weekend": None, "tok_seasonal": None}
            if hi > lo30:
                w_l, w_d, w_dow = liters[lo30:hi], dates[lo30:hi], dow[lo30:hi]
                m30 = np.nanmean(w_l) if np.isfinite(w_l).any() else np.nan
                rec["tok_use"] = use_token(m30)
                rec["tok_intensity"] = intensity_token(m30, grass)
                rec["tok_freq"] = freq_token(w_d, w_l)
                rec["tok_weekend"] = weekend_token(w_dow, w_l)
            if hi > lo180:
                rec["tok_seasonal"] = seasonal_token(dates[lo180:hi], liters[lo180:hi])
            beh_rows.append(rec)
    beh = pd.DataFrame(beh_rows)
    beh["week_start"] = pd.to_datetime(beh["week_start"])

    usage = usage.join(tok)
    usage = usage.merge(beh, on=["house_id", "week_start"], how="left")
    TOKEN_COLS = [c for c in usage.columns if c.startswith("tok_")]
    usage["keywords"] = [sorted({t for t in row if isinstance(t, str) and t})
                         for row in usage[TOKEN_COLS].to_numpy()]

    # Context string from CART bins (one token per prefix, as in the original)
    for c, p in zip(CART_TOKEN_COLS, CART_PREFIXES):
        if c not in usage:
            usage[c] = f"{p}unknown"
    usage["ctx_cart"] = usage[CART_TOKEN_COLS].fillna("unknown").astype(str).agg("|".join, axis=1)
    usage["woy"] = usage["Date"].dt.isocalendar().week.astype(int)


    # Pool top-up (rule-based; unchanged rule, flag taken from the pool_yes token)
    def pool_topup_vec(df):
        flag = df["tok_pool"].eq("pool_yes").to_numpy()
        if POOL_METHOD == "sheet":
            daily = df["Date"].map(pool_daily).fillna(0.0).to_numpy(dtype=float)
            return np.where(flag, daily, 0.0)
        if POOL_METHOD == "sheet_net":
            cd = df[["Date", "week_start", "Rainfall"]].drop_duplicates("Date").copy()
            cd["E_mm"] = cd["Date"].map(pool_daily).fillna(0.0) / POOL_AREA_M2
            cw = cd.groupby("week_start").agg(E=("E_mm", "sum"), P=("Rainfall", "sum"),
                                              ndays=("Date", "nunique"))
            cw["net_mm"] = (cw["E"] - cw["P"].fillna(0.0)).clip(lower=0)
            cw["topup_day_L"] = cw["net_mm"] * POOL_AREA_M2 / cw["ndays"]
            daily = df["week_start"].map(cw["topup_day_L"]).fillna(0.0).to_numpy()
            return np.where(flag, daily, 0.0)
        if POOL_METHOD == "rule":
            tmax_lbl = df["tok_tmax"].astype(str).str.replace("tmax_", "", regex=False)
            rain_lbl = df["tok_rain"].astype(str).str.replace("rain_", "", regex=False)
            hw = tmax_lbl.map(HEAT_FACTOR).fillna(HEAT_FACTOR["mid"])
            rw = rain_lbl.map(RAIN_FACTOR).fillna(RAIN_FACTOR["none"])
            return np.where(flag, POOL_TOPUP_HOT_NO_RAIN * hw * rw, 0.0)
        # water balance per week (climate is common to all households)
        cw = (df[["Date", "week_start", EVAP_COL, "Rainfall"]].drop_duplicates("Date")
              .groupby("week_start").agg(E=(EVAP_COL, "sum"), P=("Rainfall", "sum"),
                                         ndays=("Date", "nunique")))
        cw["net_mm"] = (cw["E"] - cw["P"]).clip(lower=0)
        cw["topup_day_L"] = cw["net_mm"] * POOL_AREA_M2 / cw["ndays"]   # 1 mm over 1 m2 = 1 L
        daily = df["week_start"].map(cw["topup_day_L"]).fillna(0.0).to_numpy()
        return np.where(flag, daily, 0.0)


    usage["pool_topup"] = pool_topup_vec(usage)
    usage["grass"] = usage["Grass Area"] if "Grass Area" in usage else np.nan
    print(f"Keywords done ({time.time() - t0:.0f}s)")

    # =====================================================================
    # 3. LDA  
    # =====================================================================
    train_texts = usage.loc[usage["is_train"], "keywords"].tolist()


    def fit_lda(seed, k=CFG["num_topics"]):
        dictionary = Dictionary(train_texts)
        dictionary.filter_extremes(no_below=CFG["no_below"], no_above=CFG["no_above"])
        corpus = [dictionary.doc2bow(t) for t in train_texts]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                       passes=CFG["passes"], random_state=seed)
        return lda, dictionary


    def infer_topics(lda, dictionary):
        keys = usage["keywords"].map(tuple)
        uniq = pd.unique(keys)                       # many documents repeat -> infer once
        k = lda.num_topics
        probs = np.zeros((len(uniq), k))
        for i, u in enumerate(uniq):
            for t, p in lda.get_document_topics(dictionary.doc2bow(list(u)), minimum_probability=0.0):
                probs[i, t] = p
        idx = pd.Index(uniq).get_indexer(keys)
        return probs[idx]


    def topic_top_words(lda, topn=10):
        return [[w for w, _ in lda.show_topic(t, topn=topn)] for t in range(lda.num_topics)]


    # =====================================================================
    # 4. BENCHMARK ENGINE
    # =====================================================================
    def loo_median(df, group_cols, val):
        """Leave-one-out median of `val` within groups; returns (median, n_peers)."""
        n_all = len(df)
        tmp = df[group_cols].copy()
        tmp["_v"] = val
        tmp["_i"] = np.arange(n_all)
        tmp = tmp[tmp["_v"].notna()].sort_values(group_cols + ["_v"], kind="mergesort")
        g = tmp.groupby(group_cols, sort=False)
        n = g["_v"].transform("size").to_numpy()
        r = g.cumcount().to_numpy()
        start = np.arange(len(tmp)) - r
        v = tmp["_v"].to_numpy()
        m = n - 1

        def kth(k):  # k-th smallest of the group excluding self
            pos = np.where(k < r, k, k + 1)
            return v[np.clip(start + pos, 0, len(v) - 1)]

        out = np.full(len(tmp), np.nan)
        odd, even = (m >= 1) & (m % 2 == 1), (m >= 2) & (m % 2 == 0)
        out[odd] = kth((m - 1) // 2)[odd]
        out[even] = 0.5 * (kth(m // 2 - 1) + kth(m // 2))[even]
        med = np.full(n_all, np.nan)
        peers = np.full(n_all, np.nan)
        med[tmp["_i"].to_numpy()] = out
        peers[tmp["_i"].to_numpy()] = m
        return med, peers


    def cf_predict(house, key, mm, is_train, seed):
        """Truncated SVD on the TRAIN house x key matrix (median mm); lookup for all rows."""
        tr = pd.DataFrame({"h": house[is_train], "k": key[is_train], "mm": mm[is_train]}).dropna()
        pivot = tr.groupby(["h", "k"])["mm"].median().unstack("k")
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            return np.full(len(house), np.nan), {}
        gm = np.nanmean(pivot.values)
        rm = pivot.mean(axis=1, skipna=True)
        cm = pivot.mean(axis=0, skipna=True)
        Xc = pivot.sub(rm, axis=0).sub(cm, axis=1).fillna(0.0)
        n_comp = max(2, min(10, min(Xc.shape) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        Z = svd.fit_transform(Xc.values)
        Xhat = Z @ svd.components_ + rm.values[:, None] + cm.values[None, :] - gm
        hi = pivot.index.get_indexer(house)
        ki = pivot.columns.get_indexer(key)
        ok = (hi >= 0) & (ki >= 0)
        out = np.full(len(house), np.nan)
        out[ok] = Xhat[hi[ok], ki[ok]]
        info = {"svd_rank": n_comp, "matrix_rows": pivot.shape[0], "matrix_cols": pivot.shape[1],
                "explained_var": float(svd.explained_variance_ratio_.sum())}
        return out, info


    def run_config(use_cart, use_lda, use_cf, use_rules, topic_main, seed):
        d = usage
        if use_cart and use_lda:
            ctx = (d["ctx_cart"] + "|" + topic_main).to_numpy()
        elif use_cart:
            ctx = d["ctx_cart"].to_numpy()
        elif use_lda:
            ctx = topic_main.to_numpy()
        else:
            ctx = np.full(len(d), "all", dtype=object)

        out = d["outdoor_liters"].to_numpy(dtype=float)
        grass = d["grass"].to_numpy(dtype=float)
        pool = d["pool_topup"].to_numpy(dtype=float) if use_rules else np.zeros(len(d))
        irr = np.clip(out - pool, 0, None)
        mm = irr / np.clip(grass, 1, None)

        gdf = pd.DataFrame({"Date": d["Date"].to_numpy(), "ctx": ctx})
        peer, n_peers = loo_median(gdf, ["Date", "ctx"], mm)

        info = {}
        if use_cf:
            key = ctx if (use_cart or use_lda) else d["woy"].to_numpy()
            cf, info = cf_predict(d["house_id"].to_numpy(), key, mm, d["is_train"].to_numpy(), seed)
            w = cf_weight(n_peers)
            base = np.where(np.isnan(peer), cf, np.where(np.isnan(cf), peer, (1 - w) * peer + w * cf))
        else:
            base = peer
        # Cold-start fallback: no peer in the context and no CF estimate ->
        # same-day leave-one-out median of ALL other households
        global_peer, _ = loo_median(pd.DataFrame({"Date": d["Date"].to_numpy(), "ctx": "all"}),
                                    ["Date", "ctx"], mm)
        used_fallback = np.isnan(base) & ~np.isnan(global_peer)
        base = np.where(np.isnan(base), global_peer, base)
        info = {**info, "fallback_share": float(np.mean(used_fallback[~np.isnan(mm)]))}
        base = np.clip(base, *BASELINE_CLIP_MM)

        lawn = np.where(np.isnan(base) | np.isnan(grass), np.nan,
                        np.where(grass > 0, base * grass / max(IRR_EFF, 1e-6), 0.0))
        rec = lawn + pool
        return {"rec": rec, "baseline_mm": base, "mm": mm, "ctx": ctx,
                "peer_mm": peer, "n_peers": n_peers, "cf_info": info}


    def weekly_eval(rec):
        d = pd.DataFrame({"house_id": usage["house_id"].to_numpy(),
                          "week_start": usage["week_start"].to_numpy(),
                          "actual": usage["outdoor_liters"].to_numpy(dtype=float),
                          "rec": rec})
        d = d[d["actual"].notna() & d["rec"].notna()]
        wk = d.groupby(["house_id", "week_start"]).agg(actual=("actual", "sum"), rec=("rec", "sum"),
                                                       ndays=("rec", "size"))
        wk = wk[wk["ndays"] >= MIN_DAYS_PER_WEEK]
        return wk[wk.index.get_level_values("week_start") >= TEST_START]


    # =====================================================================
    # 5. EXPERIMENTS
    # =====================================================================
    NAMED = {(0, 0, 0, 0): "Peer baseline",
             (0, 0, 1, 0): "CF model",
             (1, 0, 1, 1): "Hybrid model (without LDA)",
             (1, 1, 1, 1): "Proposed hybrid model (SWRS)"}
    configs = list(product([0, 1], repeat=4)) if RUN_FULL_FACTORIAL else list(NAMED.keys())


    def cfg_name(c):
        if c in NAMED:
            return NAMED[c]
        parts = [n for n, f in zip(["CART", "LDA", "CF", "Rules"], c) if f]
        return "Peer + " + " + ".join(parts) if parts else "Peer baseline"


    results = {}          # (config, seed) -> weekly test DataFrame
    topic_words = {}      # seed -> list of top words
    main_run = {}         # outputs of the main seed for exports
    cf_infos = {}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        lda, dictionary = fit_lda(seed)
        probs = infer_topics(lda, dictionary)
        topic_main = pd.Series([f"topic_{i}" for i in probs.argmax(axis=1)], index=usage.index)
        topic_words[seed] = topic_top_words(lda)
        if seed == SEEDS[0]:
            main_run["lda"], main_run["dictionary"] = lda, dictionary
            main_run["probs"], main_run["topic_main"] = probs, topic_main
        cache = {}
        for c in configs:
            cart, ldaf, cf, rules = c
            sig = (cart, ldaf, cf, rules)
            # configurations without LDA and CF do not depend on the seed
            if not ldaf and not cf and seed != SEEDS[0]:
                results[(c, seed)] = results[(c, SEEDS[0])]
                continue
            r = run_config(cart, ldaf, cf, rules, topic_main, seed)
            results[(c, seed)] = weekly_eval(r["rec"])
            cf_infos[(c, seed)] = r["cf_info"]
            if seed == SEEDS[0] and c == (1, 1, 1, 1):
                main_run["full"] = r
            print(f"  {cfg_name(c):40s} done ({time.time() - t0:.0f}s)")

    # Common evaluation set (same household-weeks for every configuration and seed)
    common = None
    for df in results.values():
        common = df.index if common is None else common.intersection(df.index)
    print(f"\nCommon test household-weeks: {len(common):,}")

    rows, abs_err = [], {}
    for c in configs:
        maes, rmses = [], []
        errs = []
        for seed in SEEDS:
            wk = results[(c, seed)].loc[common]
            e = wk["actual"] - wk["rec"]
            maes.append(np.mean(np.abs(e)))
            rmses.append(np.sqrt(np.mean(e ** 2)))
            errs.append(np.abs(e.to_numpy()))
        abs_err[c] = np.mean(errs, axis=0)
        rows.append({"Configuration": cfg_name(c), "CART": c[0], "LDA": c[1], "CF": c[2], "Rules": c[3],
                     "MAE_main_seed": maes[0], "RMSE_main_seed": rmses[0],
                     "MAE_mean": np.mean(maes), "MAE_sd": np.std(maes, ddof=1) if len(maes) > 1 else 0.0,
                     "RMSE_mean": np.mean(rmses), "RMSE_sd": np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0,
                     "coverage_weeks": int(np.mean([len(results[(c, s)]) for s in SEEDS]))})
    abl = pd.DataFrame(rows)

    # Wilcoxon signed-rank test vs. the full model (paired absolute errors)
    full = (1, 1, 1, 1)
    pvals = []
    for c in configs:
        if c == full:
            pvals.append(np.nan)
            continue
        diff = abs_err[c] - abs_err[full]
        pvals.append(wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan)
    abl["Wilcoxon_p_vs_full"] = pvals

    # Household-level (cluster) bootstrap 95% CI of MAE, main seed
    rng = np.random.default_rng(SEEDS[0])
    houses = common.get_level_values("house_id")
    uh, inv = np.unique(houses, return_inverse=True)
    boot_idx = rng.integers(0, len(uh), size=(N_BOOT, len(uh)))
    counts = np.bincount(inv, minlength=len(uh))
    lo_ci, hi_ci = [], []
    for c in configs:
        wk = results[(c, SEEDS[0])].loc[common]
        s = np.bincount(inv, weights=np.abs(wk["actual"] - wk["rec"]).to_numpy(), minlength=len(uh))
        b = s[boot_idx].sum(axis=1) / counts[boot_idx].sum(axis=1)
        lo_ci.append(np.percentile(b, 2.5))
        hi_ci.append(np.percentile(b, 97.5))
    abl["MAE_CI95_low"], abl["MAE_CI95_high"] = lo_ci, hi_ci
    abl = abl.sort_values(["CART", "LDA", "CF", "Rules"]).reset_index(drop=True)
    table2 = abl[abl["Configuration"].isin(NAMED.values())].copy()
    table2["order"] = table2["Configuration"].map({v: i for i, v in enumerate(NAMED.values())})
    table2 = table2.sort_values("order").drop(columns="order")

    # =====================================================================
    # 6. LDA DIAGNOSTICS
    # =====================================================================
    coh_rows = []
    if COHERENCE_K:
        print("\nCoherence analysis ...")
        rs = np.random.default_rng(0)
        sample = [train_texts[i] for i in rs.choice(len(train_texts), min(COHERENCE_SAMPLE, len(train_texts)), replace=False)]
        for k in COHERENCE_K:
            lda_k, dic_k = fit_lda(SEEDS[0], k)
            cv = CoherenceModel(model=lda_k, texts=sample, dictionary=dic_k, coherence="c_v", processes=1).get_coherence()
            coh_rows.append({"num_topics": k, "C_v": cv})
            print(f"  K={k}: C_v={cv:.3f}")
    coherence = pd.DataFrame(coh_rows)

    tm = main_run["topic_main"]
    prop = (tm[usage["is_train"]].value_counts(normalize=True).rename("share_train").to_frame()
            .join(tm[~usage["is_train"]].value_counts(normalize=True).rename("share_test")))
    words_main = topic_words[SEEDS[0]]
    topic_tbl = pd.DataFrame({"topic": [f"topic_{t}" for t in range(len(words_main))],
                              "top_words": [", ".join(w) for w in words_main]}).set_index("topic").join(prop)

    stab_rows = []
    for seed in SEEDS[1:]:
        ref, oth = words_main, topic_words[seed]
        J = np.array([[len(set(a) & set(b)) / len(set(a) | set(b)) for b in oth] for a in ref])
        ri, ci = linear_sum_assignment(-J)
        stab_rows.append({"seed": seed, "mean_matched_jaccard_top10": J[ri, ci].mean()})
    stability = pd.DataFrame(stab_rows)

    # =====================================================================
    # 7. MAIN-RUN OUTPUTS 
    # =====================================================================
    r = main_run["full"]
    ctx = usage[["house_id", "Date", "week_seq", "week_start", "is_train", "outdoor_liters",
                 "grass", "pool_topup"]].copy()
    ctx["keywords"] = usage["keywords"].map(" ".join)
    ctx["topic_main"] = main_run["topic_main"].to_numpy()
    ctx["context_id"] = r["ctx"]
    ctx["mm_per_m2"] = r["mm"]
    ctx["peer_mm_loo"] = r["peer_mm"]
    ctx["n_peers"] = r["n_peers"]
    ctx["baseline_mm"] = r["baseline_mm"]
    ctx["recommend_liters"] = r["rec"]
    ctx["residual_mm"] = ctx["mm_per_m2"] - ctx["baseline_mm"]
    sd = ctx.groupby(["Date", "context_id"])["residual_mm"].transform("std")
    ctx["priority_z"] = ctx["residual_mm"] / sd.replace(0, np.nan)


    def recommend_from_keywords(kset):
        recs = []
        if kset & {"rain_medium", "rain_heavy", "rain_very_heavy"}:
            recs.append("Rain today - skip lawn watering to avoid runoff.")
        elif "rain_small" in kset:
            recs.append("Light rain - reduce lawn watering duration by ~50%.")
        if ("tmax_high" in kset) or ("tmax_very_high" in kset):
            recs.append("Hot day - water early morning (before 8am) to reduce evaporation.")
        if "intensity_high" in kset:
            recs.append("High habitual intensity - use drip or water deeper but less often.")
        return recs[:3]


    ctx["recs"] = usage["keywords"].map(lambda k: recommend_from_keywords(set(k)))
    valid = ctx["outdoor_liters"].notna() & ctx["recommend_liters"].notna()
    weekly = (ctx[valid].groupby(["house_id", "week_start"], as_index=False)
              .agg(week_seq=("week_seq", "first"), actual_liters=("outdoor_liters", "sum"),
                   rec_liters=("recommend_liters", "sum"), pool_topup=("pool_topup", "sum"),
                   ndays=("recommend_liters", "size"), priority_z_mean=("priority_z", "mean"),
                   topic_week=("topic_main", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
                   grass_area=("grass", "max")))
    weekly["period"] = np.where(weekly["week_start"] >= TEST_START, "test", "train")
    wk_recs = (ctx.explode("recs").dropna(subset=["recs"]).groupby(["house_id", "week_start"])["recs"]
               .agg(lambda s: "; ".join(pd.unique(s)[:3])).rename("weekly_recs").reset_index())
    weekly = weekly.merge(wk_recs, on=["house_id", "week_start"], how="left")
    clim_week = (usage.drop_duplicates("Date").groupby("week_start").agg(Rainfall=("Rainfall", "sum"),
                                                 Min_Temperature=("Min Temperature", "mean"),
                                                 Max_Temperature=("Max Temperature", "mean"),
                                                 Soil_moisture=("Soil moisture", "mean")).reset_index())
    weekly = weekly.merge(clim_week, on="week_start", how="left")
    if EVAP_COL in usage.columns:
        ev_week = (usage[["Date", "week_start", EVAP_COL]].drop_duplicates("Date")
                   .groupby("week_start")[EVAP_COL].sum().rename("Evaporation_week_mm").reset_index())
        weekly = weekly.merge(ev_week, on="week_start", how="left")

    # Triggers: (1) YoY > 30%, (2) > 30% above 10-km peers with same topic, (3) > 30% above benchmark
    iso = weekly["week_start"].dt.isocalendar()
    weekly["iso_year"], weekly["iso_week"] = iso.year.astype(int), iso.week.astype(int)
    prev = weekly[["house_id", "iso_year", "iso_week", "actual_liters"]].copy()
    prev["iso_year"] += 1
    weekly = weekly.merge(prev.rename(columns={"actual_liters": "actual_last_year"}),
                          on=["house_id", "iso_year", "iso_week"], how="left")
    weekly["cond1_flag"] = (weekly["actual_last_year"] > 0) & (weekly["actual_liters"] > 1.3 * weekly["actual_last_year"])
    weekly["pct_over_recommendation"] = np.where(weekly["rec_liters"] > 0,
                                                 (weekly["actual_liters"] / weekly["rec_liters"] - 1) * 100, np.nan)
    weekly["cond3_flag"] = (weekly["rec_liters"] > 0) & (weekly["actual_liters"] > 1.3 * weekly["rec_liters"])

    lat_col = next((c for c in props.columns if c.strip().lower() in {"lat", "latitude"}), None)
    lon_col = next((c for c in props.columns if c.strip().lower() in {"lon", "long", "longitude"}), None)
    weekly["cond2_flag"], weekly["peer_week_median"], weekly["peer_count"] = False, np.nan, 0
    if lat_col and lon_col:
        ll = props[["house_id", lat_col, lon_col]].dropna().drop_duplicates("house_id").set_index("house_id")
        lat, lon = np.radians(ll[lat_col].to_numpy(float)), np.radians(ll[lon_col].to_numpy(float))
        dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
        dist = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
        neighbors = {h: set(ll.index[(dist[i] <= 10.0) & (np.arange(len(ll)) != i)]) for i, h in enumerate(ll.index)}
        for wk_s, g in weekly.groupby("week_start"):
            gi = g.set_index("house_id")
            for idx, row in g.iterrows():
                nb = [n for n in neighbors.get(row["house_id"], ()) if n in gi.index
                      and gi.at[n, "topic_week"] == row["topic_week"] and pd.notna(gi.at[n, "actual_liters"])]
                if len(nb) >= 2:
                    med = float(np.median(gi.loc[nb, "actual_liters"]))
                    weekly.at[idx, "peer_count"] = len(nb)
                    weekly.at[idx, "peer_week_median"] = med
                    weekly.at[idx, "cond2_flag"] = bool(med > 0 and row["actual_liters"] > 1.3 * med)
    weekly["trigger_flag"] = weekly[["cond1_flag", "cond2_flag", "cond3_flag"]].any(axis=1)

    # Table 3 (alert sensitivity, test household-weeks of the full model)
    wt = weekly[(weekly["period"] == "test") & (weekly["rec_liters"] > 0) & (weekly["ndays"] >= MIN_DAYS_PER_WEEK)]
    table3 = pd.DataFrame([{"Threshold": f"{int(t * 100)}%",
                            "Household_weeks_flagged_pct": 100 * np.mean(wt["actual_liters"] > t * wt["rec_liters"]),
                            "Households_flagged_at_least_once_pct":
                                100 * wt.loc[wt["actual_liters"] > t * wt["rec_liters"], "house_id"].nunique()
                                / max(wt["house_id"].nunique(), 1)}
                           for t in ALERT_THRESHOLDS])

    # Case-study week (Table 6/7 source): all households, week 98
    case = weekly[weekly["week_seq"] == CASE_WEEK_SEQ][["house_id", "week_start", "actual_liters", "rec_liters",
                                                         "pct_over_recommendation", "cond3_flag", "pool_topup",
                                                         "topic_week", "Rainfall"]
                + (["Evaporation_week_mm"] if "Evaporation_week_mm" in weekly.columns else [])]

    # =====================================================================
    # 8. HYPERPARAMETERS + RULES 
    # =====================================================================
    info_full = cf_infos.get(((1, 1, 1, 1), SEEDS[0]), {})
    hyper = pd.DataFrame([
        ("Split", "train", f"{usage['Date'].min().date()} to {(TEST_START - pd.Timedelta(days=1)).date()}"),
        ("Split", "test", f"{TEST_START.date()} to {usage['Date'].max().date()}"),
        ("Keywords", "behavioural window (days before target week)", LAG_SHORT),
        ("Keywords", "seasonal window (days before target week)", LAG_LONG),
        ("Keywords", "intensity cut-offs (L/m2/day)", str(CFG["intensity_cut"])),
        ("Keywords", "frequency cut-offs (days/week)", "<=1 low; <=3 med; >3 high"),
        ("Keywords", "seasonal sensitivity rule", "|early-late|/mean > 0.5 over 180 days"),
        ("CART bins", "lot / roof / grass / income / age",
         f"{CFG['lot_bins'][1:-1]} / {CFG['roof_bins'][1:-1]} / {CFG['grass_bins'][1:-1]} / "
         f"{CFG['income_bins'][1:-1]} / {CFG['age_bins'][1:-1]}"),
        ("CART bins", "use (L/day) / rain / tmax / tmin / soil",
         f"{CFG['use_bins'][1:-1]} / {CFG['rain_bins'][1:-1]} / {CFG['tmax_bins'][1:-1]} / "
         f"{CFG['tmin_bins'][1:-1]} / {CFG['soil_bins'][1:-1]}"),
        ("LDA", "num_topics", CFG["num_topics"]),
        ("LDA", "passes", CFG["passes"]),
        ("LDA", "no_below / no_above", f"{CFG['no_below']} / {CFG['no_above']}"),
        ("LDA", "alpha / eta", "gensim default (symmetric 1/K)"),
        ("LDA", "vocabulary size after filtering", len(main_run["dictionary"])),
        ("LDA", "training documents (house-days)", len(train_texts)),
        ("LDA", "seeds", str(SEEDS)),
        ("CF", "matrix", "household x context_id, median daily irrigation depth (mm), training data only"),
        ("CF", "method", "double-centred truncated SVD (row/column means), reconstruction"),
        ("CF", "rank", info_full.get("svd_rank")),
        ("CF", "matrix size", f"{info_full.get('matrix_rows')} x {info_full.get('matrix_cols')}"),
        ("CF", "explained variance (rank-k)", info_full.get("explained_var")),
        ("Blend", "w_CF by number of same-day peers", "<=2: 0.75; 3-4: 0.55; 5-9: 0.35; >=10: 0.20"),
        ("Blend", "formula", "baseline = (1-w)*peer_LOO + w*CF; fallback to the available one"),
        ("Blend", "cold start", "unseen household/context -> peer only; no peers -> CF only; "
                                "neither -> same-day leave-one-out median of all households"),
        ("Blend", "share of house-days using the all-household fallback (full model)", info_full.get("fallback_share")),
        ("Blend", "baseline clamp (mm/day)", str(BASELINE_CLIP_MM)),
        ("Rules", "irrigation efficiency", IRR_EFF),
        ("Rules", "pool top-up",
         f"daily values read from sheet '{POOL_SHEET}'" if POOL_METHOD == "sheet" else
         f"weekly max(sum(sheet '{POOL_SHEET}' / {POOL_AREA_M2}) - sum rainfall, 0) x {POOL_AREA_M2} m2"
         if POOL_METHOD == "sheet_net" else
         f"weekly max(sum evaporation - sum rainfall, 0) x {POOL_AREA_M2} m2, spread over the week"
         if POOL_METHOD == "water_balance" else
         f"{POOL_TOPUP_HOT_NO_RAIN} L/day x heat factor {HEAT_FACTOR} x rain factor {RAIN_FACTOR}"),
        ("Rules", "lawn benchmark", "baseline_mm x grass area / irrigation efficiency"),
        ("Rules", "alert", "actual weekly use > 1.3 x recommended weekly use"),
        ("Evaluation", "unit", "L/week, test household-weeks with >= %d valid days" % MIN_DAYS_PER_WEEK),
        ("Evaluation", "uncertainty", f"{len(SEEDS)} seeds (mean +/- SD); household cluster bootstrap B={N_BOOT}"),
        ("Evaluation", "significance", "Wilcoxon signed-rank on paired absolute errors vs. full model"),
    ], columns=["Component", "Setting", "Value"])

    rule_tbl = pd.DataFrame([
        ("rain_medium", "Rain today - skip lawn watering to avoid runoff."),
        ("rain_small", "Light rain - reduce lawn watering duration by ~50%."),
        ("tmax_high / tmax_very_high", "Hot day - water early morning (before 8am) to reduce evaporation."),
        ("intensity_high (30-day habitual)", "Use drip or water deeper but less often."),
        ("actual > 130% of benchmark", "Alert: reduce run-time/frequency, check leaks or pool top-ups."),
        ("actual > 130% of 10-km same-topic peer median", "Alert: peer comparison message."),
        ("actual > 130% of same week last year", "Alert: year-over-year message."),
    ], columns=["Condition", "Message / action"])

    # =====================================================================
    # 9. WRITE RESULTS
    # =====================================================================
    id_map = None
    if ANONYMISE_IDS:
        uniq_ids = sorted(usage["house_id"].unique())
        id_map = {h: f"HH{i + 1:03d}" for i, h in enumerate(uniq_ids)}
        for df_ in (ctx, weekly, case):
            df_["house_id"] = df_["house_id"].map(id_map)

    out_path = str(Path(DATA_PATH).with_name(Path(DATA_PATH).stem + "_RESULTS_v2.xlsx"))
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        table2.to_excel(w, sheet_name="Table2_main", index=False)
        abl.to_excel(w, sheet_name="Ablation_full_factorial", index=False)
        table3.to_excel(w, sheet_name="Table3_thresholds", index=False)
        case.to_excel(w, sheet_name=f"Week{CASE_WEEK_SEQ}_all_houses", index=False)
        topic_tbl.to_excel(w, sheet_name="Topic_words_proportions")
        coherence.to_excel(w, sheet_name="LDA_coherence", index=False)
        stability.to_excel(w, sheet_name="LDA_stability", index=False)
        hyper.to_excel(w, sheet_name="Hyperparameters", index=False)
        rule_tbl.to_excel(w, sheet_name="Rule_table", index=False)
        weekly.to_excel(w, sheet_name="Weekly Summary", index=False)
        ctx.drop(columns=["recs"]).to_excel(w, sheet_name="Daily_full_model", index=False)
    if id_map is not None:   # keep this file PRIVATE (do not upload)
        pd.Series(id_map, name="anon_id").to_excel(str(Path(out_path).with_name("id_mapping_PRIVATE.xlsx")))

    print("\n" + table2[["Configuration", "MAE_main_seed", "RMSE_main_seed", "MAE_mean", "MAE_sd",
                          "Wilcoxon_p_vs_full"]].round(3).to_string(index=False))
    print(f"\nWrote: {out_path}\nTotal time: {(time.time() - t0) / 60:.1f} min")



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
