# """
# Create a clean Country–Disease–Year dataset for APP regressions
# ---------------------------------------------------------------
# Extends the RQ3 pipeline to the panel level (disease × country × year),
# including full missing-data handling, imputation, and derived predictors.
#
# Output:
#   APP_country_disease_data.csv  (clean panel with imputations)
#   APP_country_disease_raw.csv   (raw merged data, before imputation)
# """
#
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
#
# # =============================================================================
# # CONFIGURATION
# # =============================================================================
#
# CUSTOM_DISEASES = [
#     'HIV/AIDS and sexually transmitted infections',
#     'Neglected tropical diseases and malaria',
#     'Maternal and neonatal disorders',
#     'Nutritional deficiencies',
#     'Respiratory infections and tuberculosis',
#     'Chronic respiratory diseases',
#     'Digestive diseases',
#     'Mental disorders',
#     'Neurological disorders',
#     'Cardiovascular diseases',
#     'Diabetes and kidney diseases',
#     'Musculoskeletal disorders',
#     'Neoplasms',
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

CUSTOM_DISEASES = [
    'HIV/AIDS and sexually transmitted infections',
    'Neglected tropical diseases and malaria',
    'Maternal and neonatal disorders',
    'Nutritional deficiencies',
    'Respiratory infections and tuberculosis',
    'Chronic respiratory diseases',
    'Digestive diseases',
    'Mental disorders',
    'Neurological disorders',
    'Cardiovascular diseases',
    'Diabetes and kidney diseases',
    'Musculoskeletal disorders',
    'Neoplasms',
    'Sense organ diseases',
    'Skin and subcutaneous diseases',
    'Substance use disorders',
    'Enteric infections'
]

# =============================================================================
# DATA LOADING - MODIFIED FOR OPEN SCIENCE
# =============================================================================

print("\n=== Loading base data (Open Science) ===")
base_dir = r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data"
aggregated_file = os.path.join(base_dir, "public_aggregated_participants_70k.csv")
gbd_file = os.path.join(base_dir, "gbddisease.csv")
all_about_country_file = os.path.join(base_dir, "AllAboutCountry.csv")
disease_mapping_file = os.path.join(base_dir, "disease_mapping.csv")

try:
    participants = pd.read_csv(aggregated_file)
    gbddisease = pd.read_csv(gbd_file)
    all_about_country = pd.read_csv(all_about_country_file, low_memory=False)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please run create_public_dataset.py first.")
    exit(1)

# Rename columns to match expected format
participants.rename(columns={"Disease": "disease_name", "YEAR": "year", "Total_Participants": "participants", "Total_Studies": "study_count"}, inplace=True)
participants = participants[participants["disease_name"].isin(CUSTOM_DISEASES)].copy()

print(f"  OK {len(participants)} country-disease-year records of trial data")

# =============================================================================
# STEP 2: Merge DALY data
# =============================================================================

print("\n=== Preparing burden (DALY) data ===")
# Need to link Disease Name to ID to link to GBD if GBD uses IDs
# But gbddisease.csv has 'cause_name'? The view showed 'cause_name'.
# Let's check headers of gbddisease... assumed from previous code 'cause_name' exists.
# Previous code: burden = gbddisease[["ISO3", "year", "cause_name", "val"]].copy()
# So it has cause_name.

burden = gbddisease[["ISO3", "year", "cause_name", "val"]].copy()
burden.rename(columns={"val": "dalys", "cause_name": "disease_name"}, inplace=True)
burden = burden[burden["disease_name"].isin(CUSTOM_DISEASES)]
print(f"  [OK] {len(burden)} country–disease–year burden records")

# =============================================================================
# STEP 3: Compute PBR
# =============================================================================

print("\n=== Calculating PBR (Participation–Burden Ratio) ===")

participants["participant_share"] = participants.groupby(
    ["disease_name", "year"]
)["participants"].transform(lambda x: x / x.sum())

burden["daly_share"] = burden.groupby(
    ["disease_name", "year"]
)["dalys"].transform(lambda x: x / x.sum())

df = participants.merge(
    burden[["ISO3", "disease_name", "year", "dalys", "daly_share"]],
    on=["ISO3", "disease_name", "year"], how="inner"
)
df["pbr"] = df["participant_share"] / (df["daly_share"] + 1e-10)
df["log_pbr"] = np.log(df["pbr"] + 0.01)

print(f"  [OK] Final panel size: {len(df)}")

# =============================================================================
# STEP 4: Merge country-level predictors
# =============================================================================


country_vars = all_about_country.pivot_table(
    index=["ISO3", "Year"],
    columns="Type",
    values="Value",
    aggfunc="first"
).reset_index()

country_vars_avg = country_vars.groupby("ISO3").agg({
    col: lambda x: pd.to_numeric(x, errors="coerce").mean()
    for col in country_vars.columns if col not in ["ISO3", "Year"]
}).reset_index()

rename_map = {
    "GDP": "gdp",
    "Population": "population",
    "HDI": "hdi",
    "Hospital beds": "hospital_beds",
    "Medical doctors (per 10,000)": "doctors_per_10k",
    "HEV": "health_expenditure",
    "RDV": "rd_expenditure",
    "TotalPub": "total_publications",
    "TotalCitation": "total_citations",
    "Hospitals": "hospitals",
    "Income": "income_group",
    "Official Development Assistance": "oda",
    "DemonIndex": "democracy_index"
}
country_vars_avg.rename(columns={k: v for k, v in rename_map.items() if k in country_vars_avg.columns}, inplace=True)
df = df.merge(country_vars_avg, on="ISO3", how="left")

# =============================================================================
# STEP 5: Derived predictors
# =============================================================================

print("\n=== Creating derived variables ===")

for col in ["gdp", "population", "health_expenditure", "total_publications"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log(df[col] + 1)

if "population" in df.columns:
    df["hospital_beds_per_capita"] = df.get("hospital_beds", 0) / df["population"] * 1000
    df["hospitals_per_capita"] = df.get("hospitals", 0) / df["population"] * 1e6
    df["health_exp_per_capita"] = df.get("health_expenditure", 0) / df["population"]
    df["log_health_exp_per_capita"] = np.log(df["health_exp_per_capita"] + 1)
    df["publications_per_capita"] = df.get("total_publications", 0) / df["population"] * 1e6
    df["log_publications_per_capita"] = np.log(df["publications_per_capita"] + 1)

# =============================================================================
# STEP 6: Missing data summary + imputation
# =============================================================================

print("\n=== Checking and imputing missing data ===")

predictors = [
    "log_gdp", "log_population", "rd_expenditure", "log_publications_per_capita",
    "total_citations", "hospital_beds_per_capita", "doctors_per_10k",
    "log_health_exp_per_capita", "hospitals_per_capita", "hdi", "democracy_index"
]

available = [p for p in predictors if p in df.columns]
for var in available:
    n_miss = df[var].isna().sum()
    pct = n_miss / len(df) * 100
    print(f"  {var:25s}: {n_miss:5d} missing ({pct:5.2f}%)")

print("\nImputing missing predictors with median values...")
imputer = SimpleImputer(strategy="median")
df[available] = imputer.fit_transform(df[available])

print("  [OK] Imputation complete")

# =============================================================================
# STEP 7: Export
# =============================================================================

df.to_csv("APP_country_disease_raw.csv", index=False)
print("  [OK] Saved raw dataset before imputation")

df.to_csv("APP_country_disease_data.csv", index=False)
print("  [OK] Final clean dataset saved: APP_country_disease_data.csv")
print(f"     Rows: {len(df)}, Countries: {df['ISO3'].nunique()}, Diseases: {df['disease_name'].nunique()}")




######## THE FOLLOWING IS TO RUN THE REGRESSION, ABOVE IS TO PREPARE FOR THE REGRESSION DATA FOR EACH DISEASE IN EACH COUNTRY

# ===========================================================
# Stage 1: individual_regression_with_residuals.py
# ===========================================================

import pandas as pd
import statsmodels.api as sm
import os

data_file = "APP_country_disease_data.csv"
output_dir = "."

dv = "log_pbr"
predictors = [
    "log_gdp",
    "rd_expenditure",
    "log_health_exp_per_capita",
    "log_publications_per_capita",
    "hdi",
    "democracy_index",
    "doctors_per_10k",
    "hospital_beds_per_capita",
]

def run_regression(df, predictors, dep_var):
    X = sm.add_constant(df[predictors])
    y = df[dep_var]
    model = sm.OLS(y, X).fit()
    results = []
    for var in predictors:
        results.append({
            "variable": var,
            "coef": model.params.get(var, float("nan")),
            "se": model.bse.get(var, float("nan")),
            "pval": model.pvalues.get(var, float("nan")),
        })
    return model, results

print("\n=== Loading dataset ===")
df = pd.read_csv(data_file)
df = df.dropna(subset=[dv] + predictors)
print(f"Dataset: {len(df):,} rows | {df['ISO3'].nunique()} countries | {df['disease_name'].nunique()} diseases")

# --- Per-disease regressions ---
print("\n=== Running per-disease regressions (saving residuals) ===")
disease_results, disease_residuals = [], []
for disease, subdf in df.groupby("disease_name"):
    if subdf["ISO3"].nunique() < 10:
        continue
    model, results = run_regression(subdf, predictors, dv)
    for r in results:
        disease_results.append({"disease": disease, "r2": model.rsquared, "n": len(subdf), **r})
    temp = subdf.copy()
    temp["predicted_log_pbr"] = model.predict(sm.add_constant(subdf[predictors]))
    temp["residual_log_pbr"] = temp[dv] - temp["predicted_log_pbr"]
    disease_residuals.append(temp)

disease_df = pd.DataFrame(disease_results)
disease_residual_df = pd.concat(disease_residuals)
disease_df.to_csv(os.path.join(output_dir, "regression_per_disease.csv"), index=False)
disease_residual_df.to_csv(os.path.join(output_dir, "residuals_per_disease.csv"), index=False)
print(f"[OK] Saved: regression_per_disease.csv ({len(disease_df)} rows)")
print(f"[OK] Saved: residuals_per_disease.csv ({len(disease_residual_df)} rows)")

# --- Per-country regressions ---
print("\n=== Running per-country regressions (saving residuals) ===")
country_results, country_residuals = [], []
for country, subdf in df.groupby("ISO3"):
    if subdf["disease_name"].nunique() < 10:
        continue
    model, results = run_regression(subdf, predictors, dv)
    for r in results:
        country_results.append({"country": country, "r2": model.rsquared, "n": len(subdf), **r})
    temp = subdf.copy()
    temp["predicted_log_pbr"] = model.predict(sm.add_constant(subdf[predictors]))
    temp["residual_log_pbr"] = temp[dv] - temp["predicted_log_pbr"]
    country_residuals.append(temp)

country_df = pd.DataFrame(country_results)
country_residual_df = pd.concat(country_residuals)
country_df.to_csv(os.path.join(output_dir, "regression_per_country.csv"), index=False)
country_residual_df.to_csv(os.path.join(output_dir, "residuals_per_country.csv"), index=False)
print(f"[OK] Saved: regression_per_country.csv ({len(country_df)} rows)")
print(f"[OK] Saved: residuals_per_country.csv ({len(country_residual_df)} rows)")

print("\n[OK] All regressions complete!  Residuals ready for APP diagnosis.")

