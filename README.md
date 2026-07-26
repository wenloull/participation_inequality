# Global Participation Inequality in Clinical Trials: Public Replication Repository

This repository contains the public aggregated datasets and analysis code for reproducing the statistical findings, figures, policy intervention simulations, and econometric models reported in the study on global clinical trial participation inequality.

---

## 📂 Repository Structure

```text
public/
├── README.md                            # Complete replication guide and dataset schema
│
├── data/                                 # Unified Public Datasets & Figure Spreadsheets (sourced from analysiswoold)
│   ├── country_macro_indicators.csv      # Macro-development & governance indicators (~20 variables)
│   ├── intervention_national_pbr_validation.csv # National PBR validation dataset (182 countries)
│   ├── figure2_panel_c_temporal_reduction.csv # Source data for Figure 2 Panel C (temporal inequality reduction)
│   ├── figure2_panel_d_theil_decomposition.csv # Source data for Figure 2 Panel D (within vs. between disease Theil decomposition)
│   ├── rq2_theil_country_grouped.csv     # Country-grouped Theil metrics (Extended Data Fig 4)
│   ├── rq2_variance_partitioning.csv     # Country vs. disease variance partitioning percentages
│   ├── intervention_statistics_national_pbr_gini.csv # Intervention Gini statistics
│   ├── scenario_full_alignment_calculated.csv # Full alignment policy intervention simulation
│   ├── scenario_targeted_alignment_calculated.csv # Targeted alignment policy intervention simulation
│   ├── network_evolution_calculated.csv # Global trial network evolution over time
│   ├── network_nodes.csv                 # Country network nodes and centralities
│   └── network_edges.csv                 # Country network collaboration edge weights
│
└── scripts/                              # 14 Sequentially Numbered Execution Scripts
    ├── 01_generate_fig1_global_overview.py
    ├── 02_generate_fig2_variance_theil.py
    ├── 03_generate_ext_fig1_enrollment_dist.py
    ├── 04_generate_ext_fig2_income_rates.py
    ├── 05_generate_ext_fig3_pbr_heatmap.py
    ├── 06_generate_ext_fig4_variance_components.py
    ├── 07_run_rq4_macro_regressions.py
    ├── 08_run_policy_interventions.py
    ├── 09_run_sensitivity_shapley.py
    ├── 10_run_non_economic_residuals.py
    ├── 11_run_alternative_ml_models.py
    ├── 12_select_best_specifications.py
    ├── 13_run_representativeness_report.py
    └── 14_verify_public_data.py
```

---

## 📊 Data Availability & Schema

All public datasets are contained within the `data/` directory:

1. **`country_macro_indicators.csv`**: Contains country-disease panel statistics and key macro development indicators:
   - **Identifiers & Core Metrics**: `ISO3`, `Disease`, `Participants`, `DALYs`, `PBR`, `log_pbr`, `income_group`
   - **Research Capacity**: `rd_expenditure`, `total_publications`, `total_citations`, `researchers_per_million`
   - **Health Infrastructure**: `health_expenditure_per_capita`, `doctors_per_10k`, `hospital_beds`, `sanitation`, `uhc_index`
   - **Governance & Development**: `gdp_per_capita`, `hdi`, `democracy_index`, `trust_scientists`, `altruism`
2. **`intervention_national_pbr_validation.csv`**: National-level aggregate participation, DALY burden, and PBR metrics across 182 countries.
3. **Spreadsheet Source Data Files**: Per-figure spreadsheets providing exact numbers behind main text and Extended Data figures.

*Note: In accordance with Nature Portfolio guidelines, proprietary per-study raw extraction datasets (`geoinfor195kwoold.csv`) are omitted from public deposition to protect primary data extractions but are available from the corresponding author upon reasonable request.*

---

## 🚀 Reproduction Workflow

### Environment Setup
Python 3.8+ is required. Install required dependencies:
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn geopandas
```

### Running the Analysis Suite
Run scripts sequentially:

```bash
# 1. Main Text Figures
python scripts/01_generate_fig1_global_overview.py
python scripts/02_generate_fig2_variance_theil.py

# 2. Extended Data Figures
python scripts/03_generate_ext_fig1_enrollment_dist.py
python scripts/04_generate_ext_fig2_income_rates.py
python scripts/05_generate_ext_fig3_pbr_heatmap.py
python scripts/06_generate_ext_fig4_variance_components.py

# 3. Macro Regressions & Interventions
python scripts/07_run_rq4_macro_regressions.py
python scripts/08_run_policy_interventions.py

# 4. Sensitivity & Machine Learning Models
python scripts/09_run_sensitivity_shapley.py
python scripts/10_run_non_economic_residuals.py
python scripts/11_run_alternative_ml_models.py
python scripts/12_select_best_specifications.py

# 5. Representativeness & Public Data Verification
python scripts/13_run_representativeness_report.py
python scripts/14_verify_public_data.py
```

---

## 📜 License
This repository is released under the **MIT License**.
