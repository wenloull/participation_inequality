import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV

# Load prepared data
from analysiswoold.sensitivity_analysis import load_and_prepare_data, OUTPUT_DIR

def main():
    print("="*80)
    print("RUNNING ALTERNATIVE METHODS: PCR, PATH ANALYSIS, RF, ELASTIC NET (analysiswoold)")
    print("="*80)

    # 1. Load and prepare data
    data, predictors = load_and_prepare_data()

    # Standardize
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    y = data_scaled['log_pbr'].values

    # Define blocks (Exactly matching the 15-Variable OLS Model)
    blocks = {
        'Economic': ['log_gdp_per_capita', 'log_population', 'foreign_aid_received'],
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }

    # =========================================================================
    # METHOD 1: PRINCIPAL COMPONENT REGRESSION (PCR)
    # =========================================================================
    print("\n--- Method 1: Principal Component Regression (PCR) ---")
    pcr_data = pd.DataFrame(index=data_scaled.index)
    loadings_list = []

    for block_name, block_vars in blocks.items():
        pca = PCA(n_components=1, random_state=42)
        X_block = data_scaled[block_vars].values
        # Ensure direction is positive for easier interpretation
        scores = pca.fit_transform(X_block)[:, 0]
        loadings = pca.components_[0]
        
        # If the average loading is negative, flip the score direction so "high score" = "high development"
        if np.mean(loadings) < 0:
            scores = -scores
            loadings = -loadings
            
        pcr_data[f'PC_{block_name}'] = scores
        
        for var, load in zip(block_vars, loadings):
            loadings_list.append({
                'Block': block_name,
                'Variable': var,
                'Loading_on_PC1': load
            })

    # Fit PCR OLS
    X_pcr = sm.add_constant(pcr_data[[f'PC_{b}' for b in blocks]])
    model_pcr = sm.OLS(y, X_pcr).fit()
    print(model_pcr.summary())

    pcr_ols_df = pd.DataFrame({
        'Variable': model_pcr.params.index,
        'Coefficient': model_pcr.params.values,
        'Std_Error': model_pcr.bse.values,
        't_value': model_pcr.tvalues.values,
        'p_value': model_pcr.pvalues.values
    })
    
    out_pcr_ols = os.path.join(OUTPUT_DIR, 'alternative_pcr_ols.csv')
    pcr_ols_df.to_csv(out_pcr_ols, index=False)
    
    loadings_df = pd.DataFrame(loadings_list)
    out_pcr_loadings = os.path.join(OUTPUT_DIR, 'alternative_pcr_loadings.csv')
    loadings_df.to_csv(out_pcr_loadings, index=False)
    print(f"  Saved PCR OLS results to: {out_pcr_ols}")
    print(f"  Saved PCR loadings to: {out_pcr_loadings}")

    # =========================================================================
    # METHOD 2: PATH ANALYSIS (SEM VIA SEQUENTIAL OLS)
    # =========================================================================
    print("\n--- Method 2: Path Analysis (Mediation SEM) ---")
    
    model_res = sm.OLS(pcr_data['PC_Research'], sm.add_constant(pcr_data[['PC_Economic', 'PC_Governance']])).fit()
    model_hea = sm.OLS(pcr_data['PC_Health'], sm.add_constant(pcr_data[['PC_Economic', 'PC_Governance']])).fit()
    model_pbr = sm.OLS(y, sm.add_constant(pcr_data[['PC_Economic', 'PC_Research', 'PC_Health', 'PC_Governance']])).fit()
    
    # Path Coefficients
    paths = {
        'Eco -> Research': model_res.params['PC_Economic'],
        'Gov -> Research': model_res.params['PC_Governance'],
        
        'Eco -> Health': model_hea.params['PC_Economic'],
        'Gov -> Health': model_hea.params['PC_Governance'],
        
        'Eco -> PBR (Direct)': model_pbr.params['PC_Economic'],
        'Res -> PBR': model_pbr.params['PC_Research'],
        'Hea -> PBR': model_pbr.params['PC_Health'],
        'Gov -> PBR (Direct)': model_pbr.params['PC_Governance']
    }
    
    # Calculate Indirect & Total Effects
    indirect_eco_res = paths['Eco -> Research'] * paths['Res -> PBR']
    indirect_eco_hea = paths['Eco -> Health'] * paths['Hea -> PBR']
    total_eco = paths['Eco -> PBR (Direct)'] + indirect_eco_res + indirect_eco_hea
    
    indirect_gov_res = paths['Gov -> Research'] * paths['Res -> PBR']
    indirect_gov_hea = paths['Gov -> Health'] * paths['Hea -> PBR']
    total_gov = paths['Gov -> PBR (Direct)'] + indirect_gov_res + indirect_gov_hea

    path_effects = [
        {'Path': 'Economic -> PBR (Direct)', 'Effect': paths['Eco -> PBR (Direct)'], 'Type': 'Direct'},
        {'Path': 'Economic -> Research -> PBR', 'Effect': indirect_eco_res, 'Type': 'Indirect'},
        {'Path': 'Economic -> Health -> PBR', 'Effect': indirect_eco_hea, 'Type': 'Indirect'},
        {'Path': 'Economic (Total Effect)', 'Effect': total_eco, 'Type': 'Total'},
        
        {'Path': 'Research -> PBR (Direct)', 'Effect': paths['Res -> PBR'], 'Type': 'Direct'},
        {'Path': 'Health -> PBR (Direct)', 'Effect': paths['Hea -> PBR'], 'Type': 'Direct'},
        
        {'Path': 'Governance -> PBR (Direct)', 'Effect': paths['Gov -> PBR (Direct)'], 'Type': 'Direct'},
        {'Path': 'Governance -> Research -> PBR', 'Effect': indirect_gov_res, 'Type': 'Indirect'},
        {'Path': 'Governance -> Health -> PBR', 'Effect': indirect_gov_hea, 'Type': 'Indirect'},
        {'Path': 'Governance (Total Effect)', 'Effect': total_gov, 'Type': 'Total'}
    ]
    
    path_df = pd.DataFrame(path_effects)
    print(path_df.to_string(index=False))
    
    out_path = os.path.join(OUTPUT_DIR, 'alternative_path_analysis.csv')
    path_df.to_csv(out_path, index=False)
    print(f"  Saved Path Analysis to: {out_path}")

    # =========================================================================
    # METHOD 3: RANDOM FOREST & PERMUTATION IMPORTANCE
    # =========================================================================
    print("\n--- Method 3: Random Forest and Permutation Importance ---")
    all_vars = [v for block_vars in blocks.values() for v in block_vars]
    X_rf = data_scaled[all_vars].values
    
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_rf, y)
    
    # Permutation importance
    result = permutation_importance(rf, X_rf, y, n_repeats=10, random_state=42)
    
    rf_df = pd.DataFrame({
        'Variable': all_vars,
        'Gini_Importance': rf.feature_importances_,
        'Permutation_Importance_Mean': result.importances_mean,
        'Permutation_Importance_Std': result.importances_std
    }).sort_values('Permutation_Importance_Mean', ascending=False)
    
    # Map back to blocks
    var_to_block = {}
    for b_name, b_vars in blocks.items():
        for v in b_vars:
            var_to_block[v] = b_name
    rf_df['Block'] = rf_df['Variable'].map(var_to_block)
    
    print(rf_df.to_string(index=False))
    
    # Block aggregation
    rf_block = rf_df.groupby('Block')[['Gini_Importance', 'Permutation_Importance_Mean']].sum().reset_index()
    rf_block = rf_block.sort_values('Permutation_Importance_Mean', ascending=False)
    print("\nBlock-Level ML Importances:")
    print(rf_block.to_string(index=False))
    
    out_rf = os.path.join(OUTPUT_DIR, 'alternative_random_forest.csv')
    rf_df.to_csv(out_rf, index=False)
    print(f"  Saved Random Forest Importances to: {out_rf}")

    # =========================================================================
    # METHOD 4: ELASTIC NET REGRESSION
    # =========================================================================
    print("\n--- Method 4: Elastic Net CV Regression ---")
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10, random_state=42)
    en.fit(X_rf, y)
    
    en_df = pd.DataFrame({
        'Variable': all_vars,
        'Block': [var_to_block[v] for v in all_vars],
        'ElasticNet_Coefficient': en.coef_
    })
    # Filter only non-zero coefficients
    en_df_nonzero = en_df[en_df['ElasticNet_Coefficient'] != 0].sort_values('ElasticNet_Coefficient', key=abs, ascending=False)
    
    print("Elastic Net Non-Zero Coefficients:")
    print(en_df_nonzero.to_string(index=False))
    
    out_en = os.path.join(OUTPUT_DIR, 'alternative_elastic_net.csv')
    en_df.to_csv(out_en, index=False)
    print(f"  Saved Elastic Net Coefficients to: {out_en}")

    print("\n" + "="*80)
    print("ALL ALTERNATIVE METHODS COMPLETED AND EXPORTED TO OUTPUT_DIR")
    print("="*80)

if __name__ == '__main__':
    main()
