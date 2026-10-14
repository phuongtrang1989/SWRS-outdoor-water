# SWRS – Smart Water Recommendation System for residential outdoor water use

Code accompanying the manuscript *"Smart Water Recommendation System (SWRS): Personalised Residential Outdoor Water Conservation through Hybrid Machine Learning and Behavioural Analytics"*.

SWRS produces weekly, household-specific **outdoor water-use benchmarks** and flags households whose use substantially exceeds that of comparable peers. It combines:

1. **CART-derived thresholds** to discretise property, socioeconomic and climatic attributes
2. **Latent Dirichlet Allocation (LDA)** on household-day keyword documents to identify behavioural topics
3. **Leave-one-out peer benchmarking + collaborative filtering (truncated SVD)**
4. **Rule-based estimation** (pool top-up, irrigation efficiency, alert rules)

## Data availability

The household smart-meter data used in the study cannot be shared because of privacy and data-sharing agreements. The folder `data/` contains a **synthetic dataset** (`sample_data_synthetic.xlsx`, 40 artificial households, 2023–2024) with exactly the same structure as the real input, so that the full workflow can be executed. Results obtained on the synthetic data will **not** reproduce the numerical values reported in the paper.

Input workbook structure:

| Sheet | Content |
|---|---|
| `property detail` | one row per household: `Property_Address` (ID), `Total Area`, `Roof Area`, `Grass Area`, `Pool`, `Age`, `Weekly Income`, `lat`, `lon` |
| `daily outdoor use` | wide format: `Property_Address` + one column per date (daily outdoor litres) |
| `Climic data` | `Date`, `Rainfall`, `Soil moisture`, `Min Temperature`, `Max Temperature`, `Evaporation` (mm/day) |
| `Weekly Pool Topup` | `Date`, `pool_topup` (daily evaporation-driven pool top-up, L/day, for a 32 m² pool) |

## Installation

```bash
pip install -r requirements.txt
```
Tested with Python 3.11–3.13.

## Usage

```bash
# run the full SWRS pipeline and experiments
python swrs_v2.py                            # or: python swrs_v2.py path/to/your_data.xlsx
# (optional) regenerate the synthetic dataset
python make_synthetic_data.py 40 data/sample_data_synthetic.xlsx
```

For a quick test run, set `SEEDS = [42]`, `RUN_FULL_FACTORIAL = False` and `COHERENCE_K = []` at the top of `swrs_v2.py`.

The CART-derived thresholds used in the paper are fixed in the `CFG` dictionary at the top of `swrs_v2.py`. Weekly pool top-up is computed as max(ΣE − ΣP, 0) × pool area, where E is evaporation and P is rainfall over the week.

## Leakage control

- Chronological split at the first week boundary on/after 1 June 2024 (train before, test after).
- Behavioural tokens use only the 30 / 180 days **before** the target week.
- The LDA dictionary/model and the CF factorisation are fitted on training data only.
- Peer benchmarks are leave-one-out (a household never contributes to its own benchmark).
- Metrics (MAE, RMSE in L/week) are computed on test household-weeks only.

## Outputs (`<input>_RESULTS_v2.xlsx`)

| Sheet | Content |
|---|---|
| `Table2_main` | main model comparison (peer baseline, CF, hybrid without LDA, full SWRS) |
| `Ablation_full_factorial` | all 16 CART × LDA × CF × Rules configurations: mean ± SD over seeds, Wilcoxon p-values, household-bootstrap 95% CI |
| `Table3_thresholds` | alert sensitivity (120 / 130 / 150%) |
| `Week98_all_houses` | case-study week: actual vs recommended use, pool top-up |
| `Topic_words_proportions`, `LDA_coherence`, `LDA_stability` | LDA diagnostics |
| `Hyperparameters`, `Rule_table` | all model settings and rules |
| `Weekly Summary`, `Daily_full_model` | detailed outputs of the full model |

Household identifiers are anonymised (HH001, …) when `ANONYMISE_IDS = True`.

## License

MIT
