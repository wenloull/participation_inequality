# Participation Inequality in Clinical Trials

This project analyzes global participation inequality in clinical trials, comparing the disease burden (DALYs) of various conditions against the volume of clinical trials (participants and study counts) conducted across different countries and income levels.

## Project Structure

The repository is organized into three main directories to separate data, analysis scripts, and utility tools.

### 📂 Directory Overview

#### `data/`
Contains all the raw and processed datasets required for the analysis.
- `gbddisease.csv`: Global Burden of Disease (GBD) data including DALYs (Disability-Adjusted Life Years).
- `AllAboutCountry.csv`: Country-level demographic and economic data (e.g., income levels, population).
- `public_aggregated_participants_70k.csv`: Aggregated trial participant data (subset of 70k).
- `public_aggregated_participants_138k.csv`: Full aggregated trial participant data (138k).
- `disease_mapping.csv`: Mapping between GBD disease IDs and user-defined disease categories.
- `rq2_bootstrap_confidence_intervals.csv`: Output data for Research Question 2 analysis.
- `geoinfor.csv`: Geographic information for mapping and visualization.

#### `scripts/`
Contains the core Python scripts used to generate figures and perform statistical analysis.
- `fig1.py`: Generates Figure 1 (Geographic maps, Heatmaps, and Income-level scatter plots).
- `fig2.py`: Generates Figure 3 (RQ2 analysis: Disease vs Country drivers of inequality).
- `create_public_dataset.py`: Aggregates raw trial data into the public datasets used by other scripts.
- `individualregression.py`: Performs regression analysis on trial participation drivers.
- `intervention.py`: Simulates interventions and generates visualization for the intervention analysis.

#### `utils/`
Helper scripts for data cleaning and system debugging.
- `fix_unicode.py`: Resolves character encoding issues in raw datasets.
- `sanitize.py`: Cleans and formats data for consistency.
- `debug_gpd.py`: Diagnostic tool for GeoPandas and Fiona environment issues.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. You can install the required libraries using:

```bash
pip install -r requirements.txt
```

### Reproducing Figures
To reproduce the primary figures from the study, run the following commands from the project root:

1. **Generate Figure 1:**
   ```bash
   python scripts/fig1.py
   ```
2. **Generate Main & SI Figure 3:**
   ```bash
   python scripts/fig2.py
   ```
3. **Generate Intervention Plots:**
   ```bash
   python scripts/intervention.py
   ```

### Data Aggregation
If you need to re-generate the aggregated public datasets from the raw files (PMIDs, years, causes):
```bash
python scripts/create_public_dataset.py
```

## Attribution
This work is part of the ongoing research on Participation Inequality in clinical trials.
Contact: Wen Lou (wlou@infor.ecnu.edu.cn)
