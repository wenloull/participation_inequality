import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define disease list
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

# Conceptual blocks for limiting factors
BLOCKS = {
    'Research_Capacity': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
    'Health_Infrastructure': ['log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation'],
    'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
}

def load_and_prepare_data():
    print("Loading and preparing updated dataset...")
    
    # Load clinical trial data and causes
    pmid_cause = pd.read_csv('/Users/wen/Desktop/participation_inequality/CauseClassier/pmid_cause.csv')
    geoinfor = pd.read_csv('/Users/wen/Desktop/participation_inequality/public/geoinfor183_disease_matched.csv')
    year_data = pd.read_csv('/Users/wen/Desktop/participation_inequality/data/year_195k.csv')
    gbddisease = pd.read_csv('/Users/wen/Desktop/participation_inequality/data/gbddisease.csv')
    all_about_country = pd.read_csv('/Users/wen/Desktop/participation_inequality/data/AllAboutCountry.csv')

    # Filter cause mapping to Level 2 and custom 16 diseases
    pmid_cause_l2 = pmid_cause[
        (pmid_cause['Level'] == 2) &
        (pmid_cause['CAUSE'].isin(CUSTOM_DISEASES))
    ].copy()

    # Merge clinical trials with geoinfor
    main_trials = year_data.merge(pmid_cause_l2, on='PMID', how='inner')
    main_trials = main_trials.merge(geoinfor[['PMID', 'Amount', 'ISO3']], on='PMID', how='inner')

    # Aggregate trials by country, disease, and year
    trial_agg = main_trials.groupby(['ISO3', 'CAUSE', 'YEAR']).agg({'Amount': 'sum'}).reset_index()
    trial_agg.rename(columns={'Amount': 'participants', 'CAUSE': 'disease_name', 'YEAR': 'year'}, inplace=True)

    # Average participants across years (2000-2024)
    part_cd = trial_agg.groupby(['ISO3', 'disease_name']).agg({'participants': 'mean'}).reset_index()

    # Prepare DALYs
    daly_df = gbddisease[['ISO3', 'year', 'cause_name', 'val']].copy()
    daly_df.rename(columns={'val': 'dalys', 'cause_name': 'disease_name'}, inplace=True)
    daly_df = daly_df[daly_df['disease_name'].isin(CUSTOM_DISEASES)]
    daly_cd = daly_df.groupby(['ISO3', 'disease_name']).agg({'dalys': 'mean'}).reset_index()

    # Merge disease-level data
    disease_merged = part_cd.merge(daly_cd, on=['ISO3', 'disease_name'], how='outer')
    disease_merged['participants'] = disease_merged['participants'].fillna(0)
    disease_merged['dalys'] = disease_merged['dalys'].fillna(0.1)

    # Calculate global totals per disease category
    global_disease_totals = disease_merged.groupby('disease_name').agg({
        'participants': 'sum',
        'dalys': 'sum'
    }).rename(columns={'participants': 'global_total_parts', 'dalys': 'global_total_dalys'}).reset_index()

    disease_merged = disease_merged.merge(global_disease_totals, on='disease_name', how='left')

    disease_merged['participant_share'] = disease_merged['participants'] / disease_merged['global_total_parts'].clip(lower=1e-6)
    disease_merged['daly_share'] = disease_merged['dalys'] / disease_merged['global_total_dalys'].clip(lower=1e-6)

    min_daly_share = 0.001
    disease_merged['adjusted_daly_share'] = np.maximum(disease_merged['daly_share'], min_daly_share)
    disease_merged['disease_pbr_ratio'] = disease_merged['participant_share'] / disease_merged['adjusted_daly_share']
    disease_merged['disease_pbr_ratio'] = np.minimum(disease_merged['disease_pbr_ratio'], 20)

    # Log PBR (adding offset to prevent log(0))
    disease_merged['log_pbr'] = np.log(disease_merged['disease_pbr_ratio'] + 1e-5)

    # Load country indicators
    country_vars = all_about_country.pivot_table(
        index=['ISO3', 'Year'],
        columns='Type',
        values='Value',
        aggfunc='first'
    ).reset_index()

    country_vars_avg = country_vars.groupby('ISO3').agg({
        col: lambda x: pd.to_numeric(x, errors='coerce').mean()
        for col in country_vars.columns if col not in ['ISO3', 'Year']
    }).reset_index()

    # Get income group mappings
    income_group_map = country_vars.groupby('ISO3')['Income'].first().to_dict()
    country_vars_avg['income_group'] = country_vars_avg['ISO3'].map(income_group_map)

    rename_map = {
        'GDP': 'gdp', 'Population': 'population', 'HDI': 'hdi',
        'Hospital beds': 'hospital_beds', 'Medical doctors (per 10,000)': 'doctors_per_10k',
        'HEV': 'health_expenditure', 'RDV': 'rd_expenditure', 'TotalPub': 'total_publications',
        'TotalCitation': 'total_citations', 'Hospitals': 'hospitals', 'DemonIndex': 'democracy_index',
        'UHC Index': 'uhc_index', 'Researchers per million': 'researchers_per_million',
        'MedSch': 'medical_schools', 'Altruism': 'altruism', 'Trust in government': 'trust_government',
        'Trust in scientists': 'trust_scientists', 'Foreign aid received (% of GNI)': 'foreign_aid_received',
        'Sanitation': 'sanitation'
    }
    country_vars_avg.rename(columns={k: v for k, v in rename_map.items() if k in country_vars_avg.columns}, inplace=True)
    country_vars_avg['gdp_per_capita'] = country_vars_avg['gdp'] / country_vars_avg['population'].clip(lower=1)
    country_vars_avg['health_exp_per_capita'] = country_vars_avg['health_expenditure'] / country_vars_avg['population'].clip(lower=1)
    country_vars_avg['publications_per_capita'] = country_vars_avg['total_publications'] / country_vars_avg['population'].clip(lower=1)

    country_vars_avg['log_population'] = np.log(country_vars_avg['population'].clip(lower=1))
    country_vars_avg['log_gdp_per_capita'] = np.log(country_vars_avg['gdp_per_capita'].clip(lower=1e-6))
    country_vars_avg['log_total_publications'] = np.log(country_vars_avg['total_publications'].clip(lower=1))
    country_vars_avg['log_total_citations'] = np.log(country_vars_avg['total_citations'].clip(lower=1))
    country_vars_avg['log_researchers_per_million'] = np.log(country_vars_avg['researchers_per_million'].clip(lower=1))
    country_vars_avg['log_medical_school'] = np.log(country_vars_avg['medical_schools'].clip(lower=1))
    country_vars_avg['log_health_expenditure_per_capita'] = np.log(country_vars_avg['health_exp_per_capita'].clip(lower=1e-6))

    predictors = [
        'log_gdp_per_capita', 'log_population', 'foreign_aid_received',
        'rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million',
        'log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation',
        'hdi', 'democracy_index', 'altruism', 'trust_scientists'
    ]
    
    # Impute missing predictor values using geo-income interacted median imputation
    mapping_df = pd.read_csv('/Users/wen/Desktop/participation_inequality/data/country_mapping_for_figure.csv')
    country_vars_avg = country_vars_avg.merge(mapping_df[['ISO3', 'Subregion']], on='ISO3', how='left')
    
    # Map clean income groups
    def clean_income(x):
        if pd.isna(x): return 'H' # TWN fallback
        x = str(x).upper()
        if 'HIGH' in x or 'H' == x: return 'H'
        if 'UPPER' in x or 'UM' in x: return 'UM'
        if 'LOWER' in x or 'LM' in x: return 'LM'
        if 'LOW' in x or 'L' == x: return 'L'
        return 'H'
    country_vars_avg['income_clean'] = country_vars_avg['income_group'].apply(clean_income)
    
    # Handle TWN income
    country_vars_avg.loc[country_vars_avg['ISO3'] == 'TWN', 'income_clean'] = 'H'
    
    # Interacted group
    country_vars_avg['geo_income_group'] = country_vars_avg['Subregion'].fillna('Unknown') + '_' + country_vars_avg['income_clean']
    
    for col in predictors:
        group_medians = country_vars_avg.groupby('geo_income_group')[col].median().to_dict()
        income_medians = country_vars_avg.groupby('income_clean')[col].median().to_dict()
        subregion_medians = country_vars_avg.groupby('Subregion')[col].median().to_dict()
        global_median = country_vars_avg[col].median()
        
        col_vals = country_vars_avg[col].values.copy()
        for idx in range(len(country_vars_avg)):
            if pd.isna(col_vals[idx]):
                grp = country_vars_avg.loc[idx, 'geo_income_group']
                inc = country_vars_avg.loc[idx, 'income_clean']
                subr = country_vars_avg.loc[idx, 'Subregion']
                
                val = group_medians.get(grp, np.nan)
                if pd.isna(val):
                    val = income_medians.get(inc, np.nan)
                if pd.isna(val):
                    val = subregion_medians.get(subr, np.nan)
                if pd.isna(val):
                    val = global_median
                col_vals[idx] = val
        country_vars_avg[col] = col_vals
        
    # Drop temp columns
    country_vars_avg.drop(columns=['Subregion', 'income_clean', 'geo_income_group'], inplace=True, errors='ignore')
    print(f"  Applied geo-income interacted median imputation to {len(predictors)} predictors")

    # Standardize predictors
    scaler = StandardScaler()
    country_vars_scaled = country_vars_avg.copy()
    country_vars_scaled[predictors] = scaler.fit_transform(country_vars_avg[predictors])

    return disease_merged, country_vars_scaled, predictors

def run_regression_and_classify(disease_merged, country_vars_scaled, predictors):
    print("Running disease-specific OLS regressions and calculating expected PBR...")
    
    # 1. Regress and calculate residuals for each disease
    results_list = []
    
    for disease in CUSTOM_DISEASES:
        sub = disease_merged[disease_merged['disease_name'] == disease].merge(country_vars_scaled, on='ISO3', how='inner')
        if len(sub) == 0:
            continue
        
        # Fit OLS ONLY on countries with positive participation for this disease!
        sub_train = sub[sub['participants'] > 0].copy()
        if len(sub_train) < 5:
            continue
            
        X = sm.add_constant(sub_train[predictors])
        y = np.log(sub_train['disease_pbr_ratio'].values)
        model = sm.OLS(y, X).fit()
        
        # Calculate predicted log_pbr and residuals for positive countries
        sub_train['predicted_log_pbr'] = model.predict(X)
        sub_train['Residual'] = y - sub_train['predicted_log_pbr']
        
        # Determine limiting factor for this disease
        b_scores = {}
        for b_name, b_vars in BLOCKS.items():
            coefs = []
            for var in b_vars:
                coef = model.params[var]
                pval = model.pvalues[var]
                if pval < 0.1:
                    coefs.append(abs(coef))
                else:
                    coefs.append(0.0)
            b_scores[b_name] = np.mean(coefs)
            
        max_b = max(b_scores.values())
        limiting_factor = 'Unknown'
        if max_b > 0:
            strong = [b for b, val in b_scores.items() if val > 0.7 * max_b]
            if len(strong) > 1:
                limiting_factor = 'Multiple_Factors'
            else:
                limiting_factor = strong[0]
        else:
            # FALLBACK: If no variables are significant, use absolute coefficients without p-value filter
            fallback_scores = {}
            for b_name, b_vars in BLOCKS.items():
                coefs = [abs(model.params[var]) for var in b_vars]
                fallback_scores[b_name] = np.mean(coefs)
            max_fb = max(fallback_scores.values())
            if max_fb > 0:
                strong = [b for b, val in fallback_scores.items() if val > 0.7 * max_fb]
                if len(strong) > 1:
                    limiting_factor = 'Multiple_Factors'
                else:
                    limiting_factor = strong[0]
            else:
                limiting_factor = 'Research_Capacity' # Global default fallback
                
        sub_train['Limiting_Factor_d'] = limiting_factor
        results_list.append(sub_train)
        
    cd_df = pd.concat(results_list, ignore_index=True)
    
    # Classify performance status using disease-specific SD threshold (+/- 0.8 * SD)
    cd_df['residual_sd'] = cd_df.groupby('disease_name')['Residual'].transform('std')
    conditions = [
        (cd_df['Residual'] > 0.9 * cd_df['residual_sd']),
        (cd_df['Residual'] < -0.5 * cd_df['residual_sd']),
        (cd_df['Residual'] >= -0.1 * cd_df['residual_sd']) & (cd_df['Residual'] <= 0.1 * cd_df['residual_sd'])
    ]
    choices = ['Over_Performing', 'Under', 'As_Expected']
    cd_df['Status'] = np.select(conditions, choices, default='Borderline')
    
    # Assign Limiting_Factor for Under
    cd_df['Limiting_Factor'] = np.where(cd_df['Status'] == 'Under', cd_df['Limiting_Factor_d'], 'As_Expected')
    cd_df['Limiting_Factor'] = np.where(cd_df['Status'] == 'Over_Performing', 'Over_Performing', cd_df['Limiting_Factor'])
    
    # Rename columns to match intervention.py expectations
    cd_df.rename(columns={
        'disease_name': 'Disease',
        'participants': 'Participants',
        'dalys': 'DALYs',
        'disease_pbr_ratio': 'PBR',
        'participant_share': 'Participant_Share',
        'daly_share': 'DALY_Share'
    }, inplace=True)
    
    # Recalculate A_norm, P_burden_norm, P_participant_norm
    cd_df['Authors'] = cd_df['total_publications'].fillna(1).astype(int)
    
    global_total_authors = cd_df.groupby('Disease')['Authors'].transform('sum')
    cd_df['A_global_share'] = cd_df['Authors'] / global_total_authors.clip(lower=1)
    cd_df['P_burden_global_share'] = cd_df['DALY_Share']
    cd_df['P_participant_global_share'] = cd_df['Participant_Share']
    
    share_sum = cd_df['A_global_share'] + cd_df['P_burden_global_share'] + cd_df['P_participant_global_share']
    cd_df['A_norm'] = cd_df['A_global_share'] / share_sum.clip(lower=1e-6)
    cd_df['P_burden_norm'] = cd_df['P_burden_global_share'] / share_sum.clip(lower=1e-6)
    cd_df['P_participant_norm'] = cd_df['P_participant_global_share'] / share_sum.clip(lower=1e-6)
    
    # Apply Matching Hierarchies for Over-Performing
    under_pairs = cd_df[cd_df['Status'] == 'Under']
    disease_modes = under_pairs.groupby('Disease')['Limiting_Factor'].agg(lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 'Research_Capacity').to_dict()
    country_modes = under_pairs.groupby('ISO3')['Limiting_Factor'].agg(lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 'Research_Capacity').to_dict()
    global_mode = under_pairs['Limiting_Factor'].mode()[0] if len(under_pairs) > 0 else 'Research_Capacity'
    
    def match_over_performing(row):
        if row['Status'] != 'Over_Performing':
            return row['Limiting_Factor']
        dis_mode = disease_modes.get(row['Disease'], 'Unknown')
        if dis_mode != 'Unknown' and dis_mode != 'As_Expected' and dis_mode != 'Over_Performing':
            return dis_mode
        cnt_mode = country_modes.get(row['ISO3'], 'Unknown')
        if cnt_mode != 'Unknown' and cnt_mode != 'As_Expected' and cnt_mode != 'Over_Performing':
            return cnt_mode
        return global_mode
        
    cd_df['Matched_Factor'] = cd_df.apply(match_over_performing, axis=1)
    
    def match_as_expected(row):
        if row['Status'] != 'As_Expected':
            return row['Matched_Factor']
        components = {'Research_Capacity': row['A_norm'], 'Health_Infrastructure': row['P_burden_norm'], 'Governance': row['P_participant_norm']}
        largest = max(components, key=components.get)
        return largest
            
    cd_df['Visual_Factor'] = cd_df.apply(lambda r: r['Matched_Factor'] if r['Status'] != 'As_Expected' else match_as_expected(r), axis=1)
    cd_df['Visual_Factor'] = np.where(cd_df['Status'] == 'Under', cd_df['Limiting_Factor'], cd_df['Visual_Factor'])
    cd_df['Visual_Factor'] = cd_df['Visual_Factor'].fillna('Unknown')
    
    # Calculate Gini function
    def calculate_gini(values):
        values = np.sort(values)
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0
        index = np.arange(1, n + 1)
        numerator = np.sum((2 * index - n - 1) * values)
        denominator = n * np.sum(values)
        return numerator / denominator
        
    # Calculate leave-one-out Gini contribution as actual CIS_Country proxy!
    calculated_cis = []
    all_countries = country_vars_scaled['ISO3'].unique()
    
    for disease in CUSTOM_DISEASES:
        sub_d = cd_df[cd_df['Disease'] == disease].copy()
        
        # Pad with zeros for all countries in the study
        full_pbr_dict = {c: 0.0 for c in all_countries}
        full_pbr_dict.update(dict(zip(sub_d['ISO3'], sub_d['PBR'])))
        full_pbr_vals = np.array([full_pbr_dict[c] for c in all_countries])
        
        base_gini = calculate_gini(full_pbr_vals)
        
        for _, row in sub_d.iterrows():
            country = row['ISO3']
            idx = list(all_countries).index(country)
            loo_vals = np.delete(full_pbr_vals, idx)
            loo_gini = calculate_gini(loo_vals)
            
            cis = (base_gini - loo_gini) / base_gini * 100 if base_gini > 0 else 0.0
            calculated_cis.append({
                'ISO3': country,
                'Disease': disease,
                'CIS_Country': cis
            })
            
    cis_df_calc = pd.DataFrame(calculated_cis)
    cd_df = cd_df.merge(cis_df_calc, on=['ISO3', 'Disease'], how='inner')
    
    cd_df['x'] = cd_df['Residual']
    cd_df['y'] = cd_df['CIS_Country']
    cd_df['CIS_Country'] = cd_df['CIS_Country'].fillna(0.0)
    
    print("\nCalculated Status counts for output:")
    print(cd_df['Status'].value_counts())
    print("\nVisual_Factor counts:")
    print(cd_df['Visual_Factor'].value_counts())
    
    return cd_df

def main():
    print("=" * 80)
    print("RUNNING INTERVENTION ANALYSIS USING ORIGINAL PLOTTING CODE IN analysis/")
    print("=" * 80)
    
    # 1. Prepare data
    disease_merged, country_vars_scaled, predictors = load_and_prepare_data()
    df = run_regression_and_classify(disease_merged, country_vars_scaled, predictors)
    
    OUTPUT_DIR = '/Users/wen/Desktop/participation_inequality/public'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save updated dataset
    df.to_csv(os.path.join(OUTPUT_DIR, 'APP_visual_factor_updated.csv'), index=False)
    
    # Add public path to search path so we can import intervention_local
    sys.path.append('/Users/wen/Desktop/participation_inequality/public')
    from intervention_local import create_network_data, create_3x2_visualization
    
    # 2. Create network using original function
    nodes_df, edges_df = create_network_data(df)
    
    # Save network files
    nodes_df.to_csv(os.path.join(OUTPUT_DIR, 'network_nodes.csv'), index=False)
    edges_df.to_csv(os.path.join(OUTPUT_DIR, 'network_edges.csv'), index=False)
    
    # 3. Change directory to output directory so all outputs are generated there
    print(f"Changing working directory to {OUTPUT_DIR} to generate original figures and CSVs...")
    os.chdir(OUTPUT_DIR)
    
    # 4. Call original create_3x2_visualization function
    create_3x2_visualization(df, nodes_df, edges_df)
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFUL GENERATION USING ORIGINAL LAYOUT AND VISUAL EFFECTS IN analysis/rq4_results/!")
    print("=" * 80)

if __name__ == "__main__":
    main()
