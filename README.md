# Global Clinical Trial Participation Inequality Analysis

## Overview
This repository contains the data and code for the analysis of global inequality in clinical trial participation. The analysis quantifies the disparity between disease burden and trial participation across countries and diseases, investigating the drivers of this inequality and potential intervention strategies.

## Dataset
The analysis relies on aggregated datasets that combine clinical trial records with Global Burden of Disease (GBD) metrics:

*   **`public_aggregated_participants_138k.csv`**: The primary aggregated dataset containing trial participation counts matched with disease burden (DALYs) for each country and disease.
*   **`APP_country_disease_data.csv`**: The processed dataset used for regression and network models, including variables such as Participation-to-Burden Ratio (PBR), GDP, and health infrastructure metrics.

## Code Structure &amp; Execution Order
The analysis pipeline consists of several Python scripts. For full reproducibility, please run them in the following order:

### 1. Data Preparation
*   **`create_public_dataset.py`**: Creates the aggregated public dataset from raw data files.
*   **Key Outputs**: `public_aggregated_participants_138k.csv`

### 2. `fig1.py`
*   **Purpose**: Generates visualizations for the geographical distribution of inequality (Figure 1), including:
    *   Geographic maps showing Participation-to-Burden Ratio (PBR) for CVD and HIV
    *   Specialization Index heatmap across countries and diseases
    *   Income-level scatter plots comparing disease burden vs. participation
*   **Key Outputs**: `combined_figure_final.png` / `.pdf`

### 3. `fig2.py`
*   **Purpose**: Generates visualizations for bivariate relationships between participation and burden (Figure 2).
*   **Key Outputs**: Figure 2 visualizations

### 4. `individualregression.py`
*   **Purpose**: Performs comprehensive regression analysis to identify country-level and disease-level predictors of participation inequality. It generates the residuals used in subsequent network analyses.
*   **Key Outputs**: 
    *   `regression_per_country.csv`
    *   `residuals_per_country.csv`
    *   `regression_per_disease.csv`
    *   `residuals_per_disease.csv`

### 5. `rq2_additional.py` / `temporaltheilrobust.py`
*   **Purpose**: Analyzes temporal trends in inequality (RQ2) and assesses the impact of hypothetical removal of top diseases/countries on global inequality metrics.
*   **Key Outputs**: 
    *   Temporal inequality trend figures
    *   Bootstrap confidence intervals

### 6. `intervention.py`
*   **Purpose**: Conducts network analysis on the residuals to identify structural barriers and simulates policy intervention scenarios to estimate their potential impact on reducing inequality.
*   **Key Outputs**: 
    *   `network_panel_D_stats.json`
    *   `scenario_full_alignment_calculated.csv`
    *   `scenario_targeted_alignment_calculated.csv`

### Utility Scripts
The `utils/` directory contains debugging and maintenance scripts:
*   `debug_gpd.py`: Tests geopandas shapefile loading
*   `fix_unicode.py`: Replaces Unicode characters with ASCII equivalents
*   `sanitize.py`: Removes non-ASCII characters from Python files

## Usage Instructions
1.  **Install Dependencies**: Ensure you have Python installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Placement**: Verify that all `.csv` data files are located in the same directory as the scripts.

3.  **Run Analysis**: Execute the scripts sequentially:
    ```bash
    python create_public_dataset.py
    python fig1.py
    python fig2.py
    python individualregression.py
    python rq2_additional.py
    python intervention.py
    ```

## Requirements
*   Python 3.8+
*   See `requirements.txt` for specific package versions.
