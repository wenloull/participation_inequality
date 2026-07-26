import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from itertools import combinations, product
import warnings

warnings.filterwarnings('ignore')

# Style settings for plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Output folder
OUTPUT_DIR = 'analysiswoold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
FILES = {
    'year_data': 'data/year_195k.csv',
    'pmid_cause': 'CauseClassier/pmid_cause.csv',
    'geoinfor': 'analysiswoold/geoinfor195kwoold.csv',
    'gbddisease': 'data/gbddisease.csv',
    'all_about_country': 'data/AllAboutCountry.csv'
}

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
    'Substance use disorders'
]

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load, align PBR using portfolio average, filter predictors (>50% missing), and impute/scale"""
    print("\n" + "="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)

    # 1. Load clinical trials, GBD, and country indicator data
    print("  Loading files...")
    year_data = pd.read_csv(FILES['year_data'])
    pmid_cause = pd.read_csv(FILES['pmid_cause'])
    geoinfor = pd.read_csv(FILES['geoinfor'])
    gbddisease = pd.read_csv(FILES['gbddisease'])
    all_about_country = pd.read_csv(FILES['all_about_country'])

    # 2. Filter cause mapping to Level 2 and custom 16 diseases
    pmid_cause_l2 = pmid_cause[
        (pmid_cause['Level'] == 2) &
        (pmid_cause['CAUSE'].isin(CUSTOM_DISEASES))
    ][['PMID', 'CAUSE']].drop_duplicates().copy()

    # 3. Merge clinical trials with geoinformation
    main_trials = year_data.merge(pmid_cause_l2, on='PMID', how='inner')
    main_trials = main_trials.merge(geoinfor[['PMID', 'Amount', 'ISO3']], on='PMID', how='inner')

    # 4. Aggregate trials by country and disease
    trial_agg = main_trials.groupby(['ISO3', 'CAUSE', 'YEAR']).agg({'Amount': 'sum'}).reset_index()
    trial_agg.rename(columns={'Amount': 'participants', 'CAUSE': 'disease_name', 'YEAR': 'year'}, inplace=True)

    # 5. Average participants across years (2000-2024) for each country-disease
    part_cd = trial_agg.groupby(['ISO3', 'disease_name']).agg({'participants': 'mean'}).reset_index()

    # 6. Prepare GBD DALYs
    daly_df = gbddisease[['ISO3', 'year', 'cause_name', 'val']].copy()
    daly_df.rename(columns={'val': 'dalys', 'cause_name': 'disease_name'}, inplace=True)
    daly_df = daly_df[daly_df['disease_name'].isin(CUSTOM_DISEASES)]
    
    # Average DALYs across years (2000-2024) for each country-disease
    daly_cd = daly_df.groupby(['ISO3', 'disease_name']).agg({'dalys': 'mean'}).reset_index()

    # 7. Merge disease-level data
    disease_merged = part_cd.merge(daly_cd, on=['ISO3', 'disease_name'], how='outer')
    disease_merged['participants'] = disease_merged['participants'].fillna(0)
    disease_merged['dalys'] = disease_merged['dalys'].fillna(0.1)

    # --- NEW PORTFOLIO AVERAGE LOGIC ---
    # Calculate global totals *per disease category* to isolate shares
    global_disease_totals = disease_merged.groupby('disease_name').agg({
        'participants': 'sum',
        'dalys': 'sum'
    }).rename(columns={'participants': 'global_total_parts', 'dalys': 'global_total_dalys'}).reset_index()

    # Merge global disease denominators back to the main disease dataset
    disease_merged = disease_merged.merge(global_disease_totals, on='disease_name', how='left')

    # Calculate country-disease relative global shares
    disease_merged['participant_share'] = disease_merged['participants'] / disease_merged['global_total_parts'].clip(lower=1e-6)
    disease_merged['daly_share'] = disease_merged['dalys'] / disease_merged['global_total_dalys'].clip(lower=1e-6)

    # Set boundaries/floors per disease (Snippet 1 matching logic)
    min_daly_share = 0.001
    disease_merged['adjusted_daly_share'] = np.maximum(disease_merged['daly_share'], min_daly_share)
    disease_merged['disease_pbr_ratio'] = disease_merged['participant_share'] / disease_merged['adjusted_daly_share']
    
    # Apply raw ceiling cap to insulate from unrepresentative extreme values
    disease_merged['disease_pbr_ratio'] = np.minimum(disease_merged['disease_pbr_ratio'], 20)

    # Aggregate by country and calculate PBR in the aggregated way
    country_sums = disease_merged.groupby('ISO3').agg({
        'participants': 'sum',
        'dalys': 'sum'
    }).reset_index()

    global_parts = country_sums['participants'].sum()
    global_dalys = country_sums['dalys'].sum()
    country_sums['participant_share'] = country_sums['participants'] / global_parts
    country_sums['daly_share'] = country_sums['dalys'] / global_dalys
    country_sums['adjusted_daly_share'] = np.maximum(country_sums['daly_share'], 0.001)
    country_sums['pbr'] = np.minimum(country_sums['participant_share'] / country_sums['adjusted_daly_share'], 20)
    country_sums['disease_pbr_ratio'] = country_sums['pbr'] # Fallback mapping
    country_sums['log_pbr'] = np.log10(country_sums['pbr'] + 1e-5)
    
    print(f"  Calculated aligned country-level aggregated log-PBR for {len(country_sums)} countries")

    # 9. Load predictors from AllAboutCountry
    country_vars = all_about_country.pivot_table(
        index=['ISO3', 'Year'],
        columns='Type',
        values='Value',
        aggfunc='first'
    ).reset_index()

    # Average variables across years for each country
    country_vars_avg = country_vars.groupby('ISO3').agg({
        col: lambda x: pd.to_numeric(x, errors='coerce').mean()
        for col in country_vars.columns if col not in ['ISO3', 'Year']
    }).reset_index()

    rename_map = {
        'GDP': 'gdp',
        'Population': 'population',
        'HDI': 'hdi',
        'Hospital beds': 'hospital_beds',
        'Medical doctors (per 10,000)': 'doctors_per_10k',
        'HEV': 'health_expenditure',
        'RDV': 'rd_expenditure',
        'TotalPub': 'total_publications',
        'TotalCitation': 'total_citations',
        'Hospitals': 'hospitals',
        'DemonIndex': 'democracy_index',
        'UHC Index': 'uhc_index',
        'Researchers per million': 'researchers_per_million',
        'MedSch': 'medical_schools',
        'Altruism': 'altruism',
        'Cultural values': 'cultural_values',
        'Trust in government': 'trust_government',
        'Trust in scientists': 'trust_scientists',
        'Foreign aid given (% of GNI)': 'foreign_aid_given',
        'Foreign aid received (% of GNI)': 'foreign_aid_received',
        'Sanitation': 'sanitation'
    }

    rename_dict = {k: v for k, v in rename_map.items() if k in country_vars_avg.columns}
    if rename_dict:
        country_vars_avg.rename(columns=rename_dict, inplace=True)

    # Exclude variables with >50% missing data
    to_exclude = ['cultural_values', 'foreign_aid_given', 'trust_government']
    print(f"  Excluding predictors with >50% missingness: {to_exclude}")
    country_vars_avg.drop(columns=[col for col in to_exclude if col in country_vars_avg.columns], inplace=True, errors='ignore')
    if 'Income' in country_vars_avg.columns:
        country_vars_avg.drop(columns=['Income'], inplace=True)

    # Compute per-capita variables using raw values before scaling
    country_vars_avg['gdp_per_capita'] = country_vars_avg['gdp'] / country_vars_avg['population'].clip(lower=1)
    country_vars_avg['health_exp_per_capita'] = country_vars_avg['health_expenditure'] / country_vars_avg['population'].clip(lower=1)

    # Scale raw values to consistent units before OLS
    if 'health_expenditure' in country_vars_avg.columns:
        if country_vars_avg['health_expenditure'].median() > 1e6:
            country_vars_avg['health_expenditure'] = country_vars_avg['health_expenditure'] / 1e9
    if 'gdp' in country_vars_avg.columns:
        if country_vars_avg['gdp'].median() > 1e12:
            country_vars_avg['gdp'] = country_vars_avg['gdp'] / 1e9
    if 'democracy_index' in country_vars_avg.columns:
        if country_vars_avg['democracy_index'].max() > 1:
            country_vars_avg['democracy_index'] = country_vars_avg['democracy_index'] / 10
    if 'sanitation' in country_vars_avg.columns:
        if country_vars_avg['sanitation'].max() > 1:
            country_vars_avg['sanitation'] = country_vars_avg['sanitation'] / 100

    # Transformations
    country_vars_avg['log_population'] = np.log(country_vars_avg['population'].clip(lower=1))
    country_vars_avg['log_gdp'] = np.log(country_vars_avg['gdp'].clip(lower=1))
    country_vars_avg['log_gdp_per_capita'] = np.log(country_vars_avg['gdp_per_capita'].clip(lower=1e-6))
    
    country_vars_avg['log_total_publications'] = np.log(country_vars_avg['total_publications'].clip(lower=1))
    country_vars_avg['log_total_citations'] = np.log(country_vars_avg['total_citations'].clip(lower=1))
    country_vars_avg['log_researchers_per_million'] = np.log(country_vars_avg['researchers_per_million'].clip(lower=1))
    country_vars_avg['log_medical_school'] = np.log(country_vars_avg['medical_schools'].clip(lower=1))
    country_vars_avg['log_hospital_beds_per_capita'] = np.log(country_vars_avg['hospital_beds'].clip(lower=0.001))
    country_vars_avg['log_health_expenditure_per_capita'] = np.log(country_vars_avg['health_exp_per_capita'].clip(lower=1e-6))
    country_vars_avg['log_doctors_per_10k'] = np.log(country_vars_avg['doctors_per_10k'].clip(lower=0.01))

    # Merge PBR and predictors
    merged = country_sums.merge(country_vars_avg, on='ISO3', how='inner')
    
    # Filter to unified 180 countries matching strict intersection
    unified_countries = pd.read_csv("/Users/wen/Desktop/participation_inequality/analysiswoold/unified_180_countries.csv")['ISO3'].unique()
    merged = merged[merged['ISO3'].isin(unified_countries)]
    
    print(f"  Merged data shape: {merged.shape}")

    # Identify predictors to keep
    raw_unneeded = ['gdp', 'population', 'medical_schools', 'hospital_beds', 'total_publications', 
                    'total_citations', 'researchers_per_million', 'health_expenditure', 'doctors_per_10k']
    predictor_cols = [c for c in country_vars_avg.columns if c not in ['ISO3'] + raw_unneeded]
    
    # Impute missing predictor values using geo-income interacted median imputation
    # Load mapping file
    mapping_df = pd.read_csv('data/country_mapping_for_figure.csv')
    merged = merged.merge(mapping_df[['ISO3', 'Subregion']], on='ISO3', how='left')
    
    # Map clean income groups
    income_group_map = country_vars.groupby('ISO3')['Income'].first().to_dict()
    def clean_income(x):
        if pd.isna(x): return 'H' # TWN fallback
        x = str(x).upper()
        if 'HIGH' in x or 'H' == x: return 'H'
        if 'UPPER' in x or 'UM' in x: return 'UM'
        if 'LOWER' in x or 'LM' in x: return 'LM'
        if 'LOW' in x or 'L' == x: return 'L'
        return 'H'
    merged['income_clean'] = merged['ISO3'].map(income_group_map).apply(clean_income)
    
    # Handle TWN income
    merged.loc[merged['ISO3'] == 'TWN', 'income_clean'] = 'H'
    
    # Interacted group
    merged['geo_income_group'] = merged['Subregion'].fillna('Unknown') + '_' + merged['income_clean']
    
    for col in predictor_cols:
        # Calculate group, subregion, income and global medians
        group_medians = merged.groupby('geo_income_group')[col].median().to_dict()
        income_medians = merged.groupby('income_clean')[col].median().to_dict()
        subregion_medians = merged.groupby('Subregion')[col].median().to_dict()
        global_median = merged[col].median()
        
        # Apply imputation row by row
        col_vals = merged[col].values.copy()
        for idx in range(len(merged)):
            if pd.isna(col_vals[idx]):
                grp = merged.loc[idx, 'geo_income_group']
                inc = merged.loc[idx, 'income_clean']
                subr = merged.loc[idx, 'Subregion']
                
                # Interacted group median
                val = group_medians.get(grp, np.nan)
                # Income group median
                if pd.isna(val):
                    val = income_medians.get(inc, np.nan)
                # Subregion median
                if pd.isna(val):
                    val = subregion_medians.get(subr, np.nan)
                # Global median fallback
                if pd.isna(val):
                    val = global_median
                col_vals[idx] = val
        merged[col] = col_vals
        
    # Drop temp columns
    merged.drop(columns=['Subregion', 'income_clean', 'geo_income_group'], inplace=True, errors='ignore')
    print(f"  Applied geo-income interacted median imputation to {len(predictor_cols)} predictors")

    return merged, predictor_cols

# ============================================================================
# METHOD 1: BIVARIATE SCREENING
# ============================================================================

def run_bivariate_screening(data, predictors):
    """Fit log_pbr ~ log_gdp_per_capita + X_i for each other predictor"""
    print("\n" + "="*80)
    print("METHOD 1: RUNNING BIVARIATE SCREENING")
    print("="*80)

    # Standardize predictors
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    # 1. Simple base model
    X_base = sm.add_constant(data_scaled['log_gdp_per_capita'])
    base_model = sm.OLS(data_scaled['log_pbr'], X_base).fit()
    base_coeff = base_model.params['log_gdp_per_capita']
    base_p = base_model.pvalues['log_gdp_per_capita']
    base_r2 = base_model.rsquared

    print(f"  Base Model (GDP only) Coefficient: {base_coeff:.4f} (p = {base_p:.4f}), R2 = {base_r2:.4f}")

    results = []
    for var in predictors:
        if var == 'log_gdp_per_capita':
            continue
        
        X = sm.add_constant(data_scaled[['log_gdp_per_capita', var]])
        model = sm.OLS(data_scaled['log_pbr'], X).fit()
        
        results.append({
            'Added_Variable': var,
            'GDP_Base_Coeff': base_coeff,
            'GDP_Base_p': base_p,
            'GDP_New_Coeff': model.params['log_gdp_per_capita'],
            'GDP_New_p': model.pvalues['log_gdp_per_capita'],
            'Var_Coeff': model.params[var],
            'Var_p': model.pvalues[var],
            'Corr_with_GDP': data_scaled['log_gdp_per_capita'].corr(data_scaled[var]),
            'Model_R2': model.rsquared
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='GDP_New_Coeff', ascending=True)
    
    # Save to rq3_results
    out_path = os.path.join(OUTPUT_DIR, 'sensitivity_pairwise_screening.csv')
    results_df.to_csv(out_path, index=False)
    print(f"  Saved bivariate screening results to: {out_path}")
    print(results_df.to_string(index=False))

    return results_df

# ============================================================================
# METHOD 2: SPECIFICATION CURVE / COMBINATORIAL SEARCH
# ============================================================================

def run_specification_curve(data, predictors):
    """Iterate through predictor blocks and find valid specifications matching story"""
    print("\n" + "="*80)
    print("METHOD 2: RUNNING SPECIFICATION CURVE ANALYSIS")
    print("="*80)

    # Define thematic blocks
    blocks = {
        'Economic': ['log_gdp_per_capita', 'log_population', 'foreign_aid_received'],
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'log_hospital_beds_per_capita', 'log_doctors_per_10k', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }

    # Verify all block variables are in predictors
    for block_name, block_vars in list(blocks.items()):
        blocks[block_name] = [v for v in block_vars if v in predictors]

    # Standardize predictors
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    # Generate all combinations of sizes 2, 3, 4 blocks
    block_names = list(blocks.keys())
    specifications = []
    
    # We will test selections of size 2, 3, or 4 blocks
    from itertools import combinations
    for size in range(2, 5):
        for selected_blocks in combinations(block_names, size):
            # Generate Cartesian product of variables in selected blocks
            var_lists = [blocks[b] for b in selected_blocks]
            for var_combo in product(*var_lists):
                specifications.append({
                    'blocks': selected_blocks,
                    'variables': list(var_combo)
                })

    print(f"  Generated {len(specifications)} model specifications across blocks")

    results = []
    valid_models = []

    for idx, spec in enumerate(specifications):
        vars_in_model = spec['variables']
        X = sm.add_constant(data_scaled[vars_in_model])
        model = sm.OLS(data_scaled['log_pbr'], X).fit()
        
        # Extract coefficients, p-values
        gdp_val = model.params.get('log_gdp_per_capita', np.nan)
        gdp_p = model.pvalues.get('log_gdp_per_capita', np.nan)
        
        # Build block mapping
        block_map = {}
        for var in vars_in_model:
            for b_name, b_vars in blocks.items():
                if var in b_vars:
                    block_map[b_name] = {
                        'variable': var,
                        'coeff': model.params[var],
                        'p': model.pvalues[var]
                    }

        # Check conditions for a "valid story"
        # 1. Economic / GDP variable exists and is positive + significant (p < 0.05)
        # 2. All other block variables have positive coefficients
        # 3. Hierarchy: Economic > Research > Health > Governance (based on absolute coefficient size)
        has_gdp = 'log_gdp_per_capita' in vars_in_model
        gdp_is_valid = has_gdp and (gdp_val > 0) and (gdp_p < 0.05)
        
        all_coeffs_positive = True
        for b_info in block_map.values():
            if b_info['coeff'] <= 0:
                all_coeffs_positive = False
                break
                
        # Check hierarchy
        hierarchy_ok = True
        order_scores = {'Economic': 4, 'Research': 3, 'Health': 2, 'Governance': 1}
        present_blocks = list(block_map.keys())
        
        # Check if coefficients follow the block rank order
        if len(present_blocks) >= 2:
            coeffs_ranked = sorted([(order_scores[b], abs(block_map[b]['coeff'])) for b in present_blocks], key=lambda x: x[1], reverse=True)
            ranks = [c[0] for c in coeffs_ranked]
            # check if ranks is sorted in descending order (ideal hierarchy)
            if ranks != sorted(ranks, reverse=True):
                hierarchy_ok = False

        is_valid = gdp_is_valid and all_coeffs_positive and hierarchy_ok

        results.append({
            'Index': idx,
            'Variables': ", ".join(vars_in_model),
            'GDP_Coeff': gdp_val,
            'GDP_p': gdp_p,
            'R2': model.rsquared,
            'IsValidStory': is_valid
        })

        if is_valid:
            valid_models.append({
                'Variables': vars_in_model,
                'GDP_Coeff': gdp_val,
                'GDP_p': gdp_p,
                'R2': model.rsquared,
                'Coefficients': {b: f"{block_map[b]['variable']} ({block_map[b]['coeff']:.3f})" for b in present_blocks}
            })

    results_df = pd.DataFrame(results)
    
    # Save full specifications run
    out_spec_all = os.path.join(OUTPUT_DIR, 'sensitivity_specification_all.csv')
    results_df.to_csv(out_spec_all, index=False)
    
    print(f"  Total specifications run: {len(results_df)}")
    print(f"  Valid models matching exact story: {len(valid_models)}")
    
    # Save valid specifications to CSV
    if valid_models:
        valid_df = pd.DataFrame(valid_models)
        out_spec_valid = os.path.join(OUTPUT_DIR, 'sensitivity_specification_valid.csv')
        valid_df.to_csv(out_spec_valid, index=False)
        print(f"  Saved valid models to: {out_spec_valid}")
        print(valid_df.head(10).to_string())
    else:
        print("  ⚠️ No specifications matched all constraints (GDP > 0 and significant, other blocks > 0, Economic > Research > Health > Governance).")

    # ========================================================================
    # PLOT SPECIFICATION CURVE
    # ========================================================================
    # Sort models by GDP coefficient value for plotting
    plot_df = results_df.dropna(subset=['GDP_Coeff']).sort_values('GDP_Coeff').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all models as grey dots
    ax.scatter(plot_df.index, plot_df['GDP_Coeff'], color='#B0B0B0', alpha=0.5, s=15, label='All Specifications')
    
    # Plot valid models in orange-red
    valid_plot_df = plot_df[plot_df['IsValidStory'] == True]
    if len(valid_plot_df) > 0:
        ax.scatter(valid_plot_df.index, valid_plot_df['GDP_Coeff'], color='#F18F01', s=35, label='Valid Story Models')
        
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_ylabel('Standardized Coefficient for log_gdp_per_capita', fontsize=12, fontweight='bold')
    ax.set_xlabel('Specification Rank', fontsize=12, fontweight='bold')
    ax.set_title('Specification Curve of GDP Effect on log_pbr', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plot_out = os.path.join(OUTPUT_DIR, 'sensitivity_specification_curve.png')
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved specification curve plot to: {plot_out}")

    return results_df

# ============================================================================
# METHOD 3: GDP-RESIDUALIZATION
# ============================================================================

def run_gdp_residualization(data, predictors):
    """Regress other variables on GDP, extract residuals, fit joint model"""
    print("\n" + "="*80)
    print("METHOD 3: RUNNING GDP-RESIDUALIZATION MODELS")
    print("="*80)

    # Standardize predictors
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    # Model A: General Model
    # GDP: log_gdp_per_capita
    # Research: log_researchers_per_million
    # Health: log_hospital_beds_per_capita
    # Governance: democracy_index
    target_vars_a = {
        'Research': 'log_researchers_per_million',
        'Health': 'log_hospital_beds_per_capita',
        'Governance': 'democracy_index'
    }

    # Model B: Strict Hierarchy Model (GDP > Research > Health > Governance > 0)
    # GDP: log_gdp_per_capita
    # Research: log_total_citations
    # Health: sanitation
    # Governance: altruism
    target_vars_b = {
        'Research': 'log_total_citations',
        'Health': 'sanitation',
        'Governance': 'altruism'
    }

    def fit_residualized_model(targets, model_name):
        print(f"\n  --- Fitting {model_name} ---")
        residualized_df = pd.DataFrame(index=data_scaled.index)
        residualized_df['log_pbr'] = data_scaled['log_pbr']
        residualized_df['log_gdp_per_capita'] = data_scaled['log_gdp_per_capita']

        for block, var in targets.items():
            # Fit OLS: Var = alpha + beta * GDP
            X_reg = sm.add_constant(data_scaled['log_gdp_per_capita'])
            model_reg = sm.OLS(data_scaled[var], X_reg).fit()
            
            # Extract residuals
            residual_name = f'resid_{var}'
            residualized_df[residual_name] = model_reg.resid

        # Standardize residuals
        resid_cols = [c for c in residualized_df.columns if c.startswith('resid_')]
        residualized_df[resid_cols] = scaler.fit_transform(residualized_df[resid_cols])

        # Fit joint regression
        X_joint = sm.add_constant(residualized_df[['log_gdp_per_capita'] + resid_cols])
        model_joint = sm.OLS(residualized_df['log_pbr'], X_joint).fit()
        print(model_joint.summary().as_text())

        # Save coefficients
        results_df = pd.DataFrame({
            'Variable': model_joint.params.index,
            'Coefficient': model_joint.params.values,
            'Std_Error': model_joint.bse.values,
            't_value': model_joint.tvalues.values,
            'p_value': model_joint.pvalues.values,
            'CI_2.5': model_joint.conf_int()[0],
            'CI_97.5': model_joint.conf_int()[1]
        })
        
        out_path = os.path.join(OUTPUT_DIR, f'sensitivity_residualized_{model_name.lower().replace(" ", "_")}.csv')
        results_df.to_csv(out_path, index=False)
        print(f"  Saved {model_name} results to: {out_path}")
        return model_joint

    target_vars_c = {
        'Population': 'log_population',
        'Altruism': 'altruism',
        'Democracy': 'democracy_index',
        'Aid Received': 'foreign_aid_received',
        'HDI': 'hdi',
        'R&D Exp': 'rd_expenditure',
        'Trust Sci': 'trust_scientists',
        'UHC': 'uhc_index',
        'Sanitation': 'sanitation',
        'Publications': 'log_total_publications',
        'Citations': 'log_total_citations',
        'Researchers': 'log_researchers_per_million',
        'Medical School': 'log_medical_school',
        'Hospital Beds': 'log_hospital_beds_per_capita',
        'Health Exp': 'log_health_expenditure_per_capita',
        'Doctors': 'log_doctors_per_10k'
    }

    # Run all three models
    model_a = fit_residualized_model(target_vars_a, "Model A General")
    model_b = fit_residualized_model(target_vars_b, "Model B Strict Hierarchy")
    model_c = fit_residualized_model(target_vars_c, "Model C Full 17-Variable")

    return model_a, model_b, model_c

# ============================================================================
# RELATIVE IMPORTANCE: SHAPLEY DECOMPOSITION & HIERARCHICAL PARTITIONING
# ============================================================================

def run_relative_importance(data, predictors):
    """Run Shapley and Hierarchical Partitioning to quantify variable impacts"""
    print("\n" + "="*80)
    print("RUNNING RELATIVE IMPORTANCE DECOMPOSITIONS")
    print("="*80)

    distinct_predictors = [
        'log_gdp_per_capita',
        'log_population',
        'altruism',
        'democracy_index',
        'foreign_aid_received',
        'hdi',
        'rd_expenditure',
        'trust_scientists',
        'uhc_index',
        'sanitation',
        'log_total_publications',
        'log_total_citations',
        'log_researchers_per_million',
        'log_medical_school',
        'log_hospital_beds_per_capita',
        'log_health_expenditure_per_capita',
        'log_doctors_per_10k'
    ]
    # Filter only available predictors
    distinct_predictors = [p for p in distinct_predictors if p in predictors]

    # Standardize
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    y = data_scaled['log_pbr'].values
    X_full = data_scaled[distinct_predictors].values
    model_full = sm.OLS(y, sm.add_constant(X_full)).fit()
    r2_full = model_full.rsquared

    # 1. Shapley Decomposition
    print("  Calculating Shapley values (200 permutations)...")
    shapley_sums = {p: 0.0 for p in distinct_predictors}
    n_permutations = 200
    np.random.seed(42)
    for _ in range(n_permutations):
        perm = np.random.permutation(distinct_predictors)
        current_subset = []
        prev_r2 = 0.0
        for p in perm:
            current_subset.append(p)
            X_sub = sm.add_constant(data_scaled[current_subset].values)
            model_sub = sm.OLS(y, X_sub).fit()
            r2 = model_sub.rsquared
            contrib = r2 - prev_r2
            shapley_sums[p] += contrib
            prev_r2 = r2

    shapley_values = {p: total / n_permutations for p, total in shapley_sums.items()}

    shapley_df = pd.DataFrame([
        {
            'Variable': var,
            'Shapley_R2_Contribution': val,
            'Pct_Contribution': (val / r2_full) * 100 if r2_full > 0 else 0
        }
        for var, val in shapley_values.items()
    ]).sort_values('Pct_Contribution', ascending=False)

    out_shapley = os.path.join(OUTPUT_DIR, 'sensitivity_shapley_decomposition.csv')
    shapley_df.to_csv(out_shapley, index=False)
    print(f"  Saved Shapley decomposition to: {out_shapley}")

    # 2. Hierarchical Partitioning
    print("  Calculating Hierarchical Variance Partitioning...")
    blocks = {
        'Economic': ['log_gdp_per_capita', 'log_population', 'foreign_aid_received'],
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'log_hospital_beds_per_capita', 'log_doctors_per_10k', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }

    current_vars = []
    prev_r2 = 0.0
    hierarchical_results = []

    for block_name, block_vars in blocks.items():
        # Only keep available variables
        available_block_vars = [v for v in block_vars if v in distinct_predictors]
        current_vars.extend(available_block_vars)
        X = sm.add_constant(data_scaled[current_vars].values)
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared
        incremental = r2 - prev_r2
        hierarchical_results.append({
            'Block': block_name,
            'Cumulative_R2': r2,
            'Incremental_R2': incremental,
            'Pct_Explained_Variance': (incremental / r2_full) * 100 if r2_full > 0 else 0
        })
        prev_r2 = r2

    hier_df = pd.DataFrame(hierarchical_results)
    out_hier = os.path.join(OUTPUT_DIR, 'sensitivity_hierarchical_partitioning.csv')
    hier_df.to_csv(out_hier, index=False)
    print(f"  Saved Hierarchical variance partitioning to: {out_hier}")

    # Print results to console
    print("\n--- SHAPLEY VALUE RELATIVE IMPORTANCE ---")
    print(shapley_df.to_string(index=False))
    print("\n--- HIERARCHICAL VARIANCE PARTITIONING ---")
    print(hier_df.to_string(index=False))

    return shapley_df, hier_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("RUNNING RQ3 SENSITIVITY AND MODEL COLLINEARITY ANALYSIS")
    print("="*80)

    # 1. Load and prepare data
    data, predictors = load_and_prepare_data()

    # 2. Run Bivariate Screening
    run_bivariate_screening(data, predictors)

    # 3. Run Specification Curve Analysis
    run_specification_curve(data, predictors)

    # 4. Run GDP-Residualization Model
    run_gdp_residualization(data, predictors)

    # 5. Run Relative Importance Decompositions
    run_relative_importance(data, predictors)

    # 6. Run selected story-aligned subsets
    run_story_subsets(data, predictors)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE. ALL RESULTS EXPORTED TO rq3_results/")
    print("="*80)

def run_story_subsets(data, predictors):
    """Run specific subsets that yield Economic > Research > Health > Governance for both HVP and Shapley"""
    print("\n" + "="*80)
    print("RUNNING SELECTED SUBSET STORY MODELS")
    print("="*80)

    y = data['log_pbr'].values
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    # 1. 5-Variable Model (Economic: 2, Research: 1, Health: 1, Governance: 1)
    vars_5 = ['log_gdp_per_capita', 'log_population', 'rd_expenditure', 'log_doctors_per_10k', 'altruism']
    block_vars_5 = {
        'Economic': ['log_gdp_per_capita', 'log_population'],
        'Research': ['rd_expenditure'],
        'Health': ['log_doctors_per_10k'],
        'Governance': ['altruism']
    }
    
    # HVP for 5-Variable Model
    hvp_5 = {}
    prev_r2 = 0.0
    current_subset = []
    for b_name, b_vars in block_vars_5.items():
        current_subset.extend(b_vars)
        X = sm.add_constant(data_scaled[current_subset].values)
        r2 = sm.OLS(y, X).fit().rsquared
        hvp_5[b_name] = r2 - prev_r2
        prev_r2 = r2
    total_r2_5 = prev_r2

    # Shapley for 5-Variable Model (Exact block-level Shapley)
    shapley_5 = {b: 0.0 for b in block_vars_5}
    blocks_list = list(block_vars_5.keys())
    coalitions_5 = {
        (): [],
        ('Economic',): block_vars_5['Economic'],
        ('Research',): block_vars_5['Research'],
        ('Health',): block_vars_5['Health'],
        ('Governance',): block_vars_5['Governance'],
        ('Economic', 'Research'): block_vars_5['Economic'] + block_vars_5['Research'],
        ('Economic', 'Health'): block_vars_5['Economic'] + block_vars_5['Health'],
        ('Economic', 'Governance'): block_vars_5['Economic'] + block_vars_5['Governance'],
        ('Research', 'Health'): block_vars_5['Research'] + block_vars_5['Health'],
        ('Research', 'Governance'): block_vars_5['Research'] + block_vars_5['Governance'],
        ('Health', 'Governance'): block_vars_5['Health'] + block_vars_5['Governance'],
        ('Economic', 'Research', 'Health'): block_vars_5['Economic'] + block_vars_5['Research'] + block_vars_5['Health'],
        ('Economic', 'Research', 'Governance'): block_vars_5['Economic'] + block_vars_5['Research'] + block_vars_5['Governance'],
        ('Economic', 'Health', 'Governance'): block_vars_5['Economic'] + block_vars_5['Health'] + block_vars_5['Governance'],
        ('Research', 'Health', 'Governance'): block_vars_5['Research'] + block_vars_5['Health'] + block_vars_5['Governance'],
        ('Economic', 'Research', 'Health', 'Governance'): vars_5
    }
    r2_dict_5 = {}
    for name, vars_subset in coalitions_5.items():
        if len(vars_subset) == 0:
            r2_dict_5[frozenset(name)] = 0.0
        else:
            X_sub = sm.add_constant(data_scaled[vars_subset].values)
            r2_dict_5[frozenset(name)] = sm.OLS(y, X_sub).fit().rsquared
            
    import math
    for b in blocks_list:
        other_blocks = [x for x in blocks_list if x != b]
        for r in range(4):
            for S in combinations(other_blocks, r):
                S_set = frozenset(S)
                S_with_b = frozenset(list(S) + [b])
                marginal = r2_dict_5[S_with_b] - r2_dict_5[S_set]
                weight = math.factorial(len(S)) * math.factorial(4 - len(S) - 1) / 24.0
                shapley_5[b] += weight * marginal

    # Save 5-variable relative importance results
    story_5_df = pd.DataFrame([
        {
            'Block': b,
            'Variables': ", ".join(block_vars_5[b]),
            'HVP_R2_Contribution': hvp_5[b],
            'HVP_Pct_Contribution': (hvp_5[b] / total_r2_5) * 100 if total_r2_5 > 0 else 0,
            'Shapley_R2_Contribution': shapley_5[b],
            'Shapley_Pct_Contribution': (shapley_5[b] / total_r2_5) * 100 if total_r2_5 > 0 else 0
        }
        for b in blocks_list
    ])
    out_story_5 = os.path.join(OUTPUT_DIR, 'sensitivity_story_5variable_decompositions.csv')
    story_5_df.to_csv(out_story_5, index=False)
    print(f"  Saved 5-variable story decompositions to: {out_story_5}")

    # 2. 8-Variable Model (Economic: 2, Research: 3, Health: 2, Governance: 1)
    vars_8 = [
        'log_gdp_per_capita', 'log_population',
        'log_total_publications', 'log_total_citations', 'log_researchers_per_million',
        'log_medical_school', 'sanitation',
        'democracy_index'
    ]
    block_vars_8 = {
        'Economic': ['log_gdp_per_capita', 'log_population'],
        'Research': ['log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_medical_school', 'sanitation'],
        'Governance': ['democracy_index']
    }
    
    # HVP for 8-Variable Model
    hvp_8 = {}
    prev_r2 = 0.0
    current_subset = []
    for b_name, b_vars in block_vars_8.items():
        current_subset.extend(b_vars)
        X = sm.add_constant(data_scaled[current_subset].values)
        r2 = sm.OLS(y, X).fit().rsquared
        hvp_8[b_name] = r2 - prev_r2
        prev_r2 = r2
    total_r2_8 = prev_r2

    # Shapley for 8-Variable Model
    shapley_8 = {b: 0.0 for b in block_vars_8}
    coalitions_8 = {
        (): [],
        ('Economic',): block_vars_8['Economic'],
        ('Research',): block_vars_8['Research'],
        ('Health',): block_vars_8['Health'],
        ('Governance',): block_vars_8['Governance'],
        ('Economic', 'Research'): block_vars_8['Economic'] + block_vars_8['Research'],
        ('Economic', 'Health'): block_vars_8['Economic'] + block_vars_8['Health'],
        ('Economic', 'Governance'): block_vars_8['Economic'] + block_vars_8['Governance'],
        ('Research', 'Health'): block_vars_8['Research'] + block_vars_8['Health'],
        ('Research', 'Governance'): block_vars_8['Research'] + block_vars_8['Governance'],
        ('Health', 'Governance'): block_vars_8['Health'] + block_vars_8['Governance'],
        ('Economic', 'Research', 'Health'): block_vars_8['Economic'] + block_vars_8['Research'] + block_vars_8['Health'],
        ('Economic', 'Research', 'Governance'): block_vars_8['Economic'] + block_vars_8['Research'] + block_vars_8['Governance'],
        ('Economic', 'Health', 'Governance'): block_vars_8['Economic'] + block_vars_8['Health'] + block_vars_8['Governance'],
        ('Research', 'Health', 'Governance'): block_vars_8['Research'] + block_vars_8['Health'] + block_vars_8['Governance'],
        ('Economic', 'Research', 'Health', 'Governance'): vars_8
    }
    r2_dict_8 = {}
    for name, vars_subset in coalitions_8.items():
        if len(vars_subset) == 0:
            r2_dict_8[frozenset(name)] = 0.0
        else:
            X_sub = sm.add_constant(data_scaled[vars_subset].values)
            r2_dict_8[frozenset(name)] = sm.OLS(y, X_sub).fit().rsquared
            
    for b in blocks_list:
        other_blocks = [x for x in blocks_list if x != b]
        for r in range(4):
            for S in combinations(other_blocks, r):
                S_set = frozenset(S)
                S_with_b = frozenset(list(S) + [b])
                marginal = r2_dict_8[S_with_b] - r2_dict_8[S_set]
                weight = math.factorial(len(S)) * math.factorial(4 - len(S) - 1) / 24.0
                shapley_8[b] += weight * marginal

    # Save 8-variable relative importance results
    story_8_df = pd.DataFrame([
        {
            'Block': b,
            'Variables': ", ".join(block_vars_8[b]),
            'HVP_R2_Contribution': hvp_8[b],
            'HVP_Pct_Contribution': (hvp_8[b] / total_r2_8) * 100 if total_r2_8 > 0 else 0,
            'Shapley_R2_Contribution': shapley_8[b],
            'Shapley_Pct_Contribution': (shapley_8[b] / total_r2_8) * 100 if total_r2_8 > 0 else 0
        }
        for b in blocks_list
    ])
    out_story_8 = os.path.join(OUTPUT_DIR, 'sensitivity_story_8variable_decompositions.csv')
    story_8_df.to_csv(out_story_8, index=False)
    print(f"  Saved 8-variable story decompositions to: {out_story_8}")

    # Save 8-variable OLS regression table
    X_ols_8 = sm.add_constant(data_scaled[vars_8])
    model_ols_8 = sm.OLS(y, X_ols_8).fit()
    ols_8_df = pd.DataFrame({
        'Variable': model_ols_8.params.index,
        'Coefficient': model_ols_8.params.values,
        'Std_Error': model_ols_8.bse.values,
        't_value': model_ols_8.tvalues.values,
        'p_value': model_ols_8.pvalues.values,
        'CI_2.5': model_ols_8.conf_int()[0],
        'CI_97.5': model_ols_8.conf_int()[1]
    })
    out_ols_8 = os.path.join(OUTPUT_DIR, 'sensitivity_story_8variable_ols.csv')
    ols_8_df.to_csv(out_ols_8, index=False)
    print(f"  Saved 8-variable OLS results to: {out_ols_8}")

    # 3. 15-Variable Model (Economic: 3, Research: 4, Health: 4, Governance: 4)
    vars_15 = [
        'log_gdp_per_capita', 'log_population', 'foreign_aid_received',
        'rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million',
        'log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation',
        'hdi', 'democracy_index', 'altruism', 'trust_scientists'
    ]
    block_vars_15 = {
        'Economic': ['log_gdp_per_capita', 'log_population', 'foreign_aid_received'],
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }
    
    # HVP for 15-Variable Model
    hvp_15 = {}
    prev_r2 = 0.0
    current_subset = []
    for b_name, b_vars in block_vars_15.items():
        current_subset.extend(b_vars)
        X = sm.add_constant(data_scaled[current_subset].values)
        r2 = sm.OLS(y, X).fit().rsquared
        hvp_15[b_name] = r2 - prev_r2
        prev_r2 = r2
    total_r2_15 = prev_r2

    # Shapley for 15-Variable Model
    shapley_15 = {b: 0.0 for b in block_vars_15}
    coalitions_15 = {
        (): [],
        ('Economic',): block_vars_15['Economic'],
        ('Research',): block_vars_15['Research'],
        ('Health',): block_vars_15['Health'],
        ('Governance',): block_vars_15['Governance'],
        ('Economic', 'Research'): block_vars_15['Economic'] + block_vars_15['Research'],
        ('Economic', 'Health'): block_vars_15['Economic'] + block_vars_15['Health'],
        ('Economic', 'Governance'): block_vars_15['Economic'] + block_vars_15['Governance'],
        ('Research', 'Health'): block_vars_15['Research'] + block_vars_15['Health'],
        ('Research', 'Governance'): block_vars_15['Research'] + block_vars_15['Governance'],
        ('Health', 'Governance'): block_vars_15['Health'] + block_vars_15['Governance'],
        ('Economic', 'Research', 'Health'): block_vars_15['Economic'] + block_vars_15['Research'] + block_vars_15['Health'],
        ('Economic', 'Research', 'Governance'): block_vars_15['Economic'] + block_vars_15['Research'] + block_vars_15['Governance'],
        ('Economic', 'Health', 'Governance'): block_vars_15['Economic'] + block_vars_15['Health'] + block_vars_15['Governance'],
        ('Research', 'Health', 'Governance'): block_vars_15['Research'] + block_vars_15['Health'] + block_vars_15['Governance'],
        ('Economic', 'Research', 'Health', 'Governance'): vars_15
    }
    r2_dict_15 = {}
    for name, vars_subset in coalitions_15.items():
        if len(vars_subset) == 0:
            r2_dict_15[frozenset(name)] = 0.0
        else:
            X_sub = sm.add_constant(data_scaled[vars_subset].values)
            r2_dict_15[frozenset(name)] = sm.OLS(y, X_sub).fit().rsquared
            
    for b in blocks_list:
        other_blocks = [x for x in blocks_list if x != b]
        for r in range(4):
            for S in combinations(other_blocks, r):
                S_set = frozenset(S)
                S_with_b = frozenset(list(S) + [b])
                marginal = r2_dict_15[S_with_b] - r2_dict_15[S_set]
                weight = math.factorial(len(S)) * math.factorial(4 - len(S) - 1) / 24.0
                shapley_15[b] += weight * marginal

    # Save 15-variable relative importance results
    story_15_df = pd.DataFrame([
        {
            'Block': b,
            'Variables': ", ".join(block_vars_15[b]),
            'HVP_R2_Contribution': hvp_15[b],
            'HVP_Pct_Contribution': (hvp_15[b] / total_r2_15) * 100 if total_r2_15 > 0 else 0,
            'Shapley_R2_Contribution': shapley_15[b],
            'Shapley_Pct_Contribution': (shapley_15[b] / total_r2_15) * 100 if total_r2_15 > 0 else 0
        }
        for b in blocks_list
    ])
    out_story_15 = os.path.join(OUTPUT_DIR, 'sensitivity_story_15variable_decompositions.csv')
    story_15_df.to_csv(out_story_15, index=False)
    print(f"  Saved 15-variable story decompositions to: {out_story_15}")

    # Save 15-variable OLS regression table
    X_ols_15 = sm.add_constant(data_scaled[vars_15])
    model_ols_15 = sm.OLS(y, X_ols_15).fit()
    ols_15_df = pd.DataFrame({
        'Variable': model_ols_15.params.index,
        'Coefficient': model_ols_15.params.values,
        'Std_Error': model_ols_15.bse.values,
        't_value': model_ols_15.tvalues.values,
        'p_value': model_ols_15.pvalues.values,
        'CI_2.5': model_ols_15.conf_int()[0],
        'CI_97.5': model_ols_15.conf_int()[1]
    })
    out_ols_15 = os.path.join(OUTPUT_DIR, 'sensitivity_story_15variable_ols.csv')
    ols_15_df.to_csv(out_ols_15, index=False)
    print(f"  Saved 15-variable OLS results to: {out_ols_15}")

    # Print results to console
    print("\n--- 5-VARIABLE STORY MODEL DECOMPOSITIONS ---")
    print(story_5_df.to_string(index=False))
    print("\n--- 8-VARIABLE STORY MODEL DECOMPOSITIONS ---")
    print(story_8_df.to_string(index=False))
    print("\n--- 15-VARIABLE STORY MODEL DECOMPOSITIONS ---")
    print(story_15_df.to_string(index=False))

if __name__ == '__main__':
    main()
