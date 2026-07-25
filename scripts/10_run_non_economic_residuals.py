import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from itertools import combinations

# Load prepared data
from sensitivity_analysis import load_and_prepare_data, OUTPUT_DIR

def main():
    print("="*80)
    print("RUNNING PART 2: RESIDUAL ANALYSIS EXCLUDING ECONOMIC BLOCK (OPTION B) (analysiswoold)")
    print("="*80)

    # 1. Load and prepare data
    data, predictors = load_and_prepare_data()

    # Standardize
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    y_raw = data_scaled['log_pbr'].values

    # Define blocks
    econ_vars = ['log_gdp_per_capita', 'log_population', 'foreign_aid_received']
    blocks = {
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }
    all_vars = [v for block_vars in blocks.values() for v in block_vars]

    # -------------------------------------------------------------------------
    # STEP A: FIT BASE ECONOMIC MODEL AND EXTRACT RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- Step A: Regressing PBR on Economic Block ---")
    X_econ = sm.add_constant(data_scaled[econ_vars])
    model_econ = sm.OLS(y_raw, X_econ).fit()
    print(model_econ.summary())
    
    # Residuals represent unexplained variance after controlling for wealth and population
    y_res = model_econ.resid
    print(f"  Calculated residual log-PBR (N={len(y_res)})")

    # -------------------------------------------------------------------------
    # STEP B: FIT FULL 12-VARIABLE OLS MODEL ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- OLS Regression on Residuals (12 Variables) ---")
    X_ols = sm.add_constant(data_scaled[all_vars])
    model_ols = sm.OLS(y_res, X_ols).fit()
    print(model_ols.summary())
    
    ols_df = pd.DataFrame({
        'Variable': model_ols.params.index,
        'Coefficient': model_ols.params.values,
        'Std_Error': model_ols.bse.values,
        't_value': model_ols.tvalues.values,
        'p_value': model_ols.pvalues.values,
        'CI_2.5': model_ols.conf_int()[0],
        'CI_97.5': model_ols.conf_int()[1]
    })
    out_ols = os.path.join(OUTPUT_DIR, 'no_economic_story_ols.csv')
    ols_df.to_csv(out_ols, index=False)
    print(f"  Saved OLS results to: {out_ols}")

    total_r2 = model_ols.rsquared

    # -------------------------------------------------------------------------
    # STEP C: HIERARCHICAL VARIANCE PARTITIONING (HVP: Research -> Health -> Governance)
    # -------------------------------------------------------------------------
    print("\n--- Hierarchical Variance Partitioning (HVP) on Residuals ---")
    hvp = {}
    prev_r2 = 0.0
    current_subset = []
    blocks_list = ['Research', 'Health', 'Governance']
    
    for b_name in blocks_list:
        current_subset.extend(blocks[b_name])
        X_sub = sm.add_constant(data_scaled[current_subset].values)
        r2 = sm.OLS(y_res, X_sub).fit().rsquared
        hvp[b_name] = r2 - prev_r2
        prev_r2 = r2

    # -------------------------------------------------------------------------
    # STEP D: BLOCK-LEVEL SHAPLEY DECOMPOSITION ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- Block-Level Shapley Decomposition on Residuals ---")
    coalitions = {
        (): [],
        ('Research',): blocks['Research'],
        ('Health',): blocks['Health'],
        ('Governance',): blocks['Governance'],
        ('Research', 'Health'): blocks['Research'] + blocks['Health'],
        ('Research', 'Governance'): blocks['Research'] + blocks['Governance'],
        ('Health', 'Governance'): blocks['Health'] + blocks['Governance'],
        ('Research', 'Health', 'Governance'): all_vars
    }
    
    r2_dict = {}
    for name, vars_subset in coalitions.items():
        if len(vars_subset) == 0:
            r2_dict[frozenset(name)] = 0.0
        else:
            X_sub = sm.add_constant(data_scaled[vars_subset].values)
            r2_dict[frozenset(name)] = sm.OLS(y_res, X_sub).fit().rsquared
            
    shapley = {b: 0.0 for b in blocks_list}
    for b in blocks_list:
        other_blocks = [x for x in blocks_list if x != b]
        for r in range(3): # |S| from 0 to 2
            for S in combinations(other_blocks, r):
                S_set = frozenset(S)
                S_with_b = frozenset(list(S) + [b])
                marginal = r2_dict[S_with_b] - r2_dict[S_set]
                weight = math.factorial(len(S)) * math.factorial(3 - len(S) - 1) / 6.0
                shapley[b] += weight * marginal

    # Save relative importance decompositions
    decomp_df = pd.DataFrame([
        {
            'Block': b,
            'Variables': ", ".join(blocks[b]),
            'HVP_R2_Contribution': hvp[b],
            'HVP_Pct_Contribution': (hvp[b] / total_r2) * 100 if total_r2 > 0 else 0,
            'Shapley_R2_Contribution': shapley[b],
            'Shapley_Pct_Contribution': (shapley[b] / total_r2) * 100 if total_r2 > 0 else 0
        }
        for b in blocks_list
    ])
    out_decomp = os.path.join(OUTPUT_DIR, 'no_economic_story_decompositions.csv')
    decomp_df.to_csv(out_decomp, index=False)
    print(f"  Saved decompositions to: {out_decomp}")
    print(decomp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP E: METHOD 1: PRINCIPAL COMPONENT REGRESSION (PCR) ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- PCR (Excluding Economic) on Residuals ---")
    pcr_data = pd.DataFrame(index=data_scaled.index)
    loadings_list = []

    for block_name, block_vars in blocks.items():
        pca = PCA(n_components=1, random_state=42)
        X_block = data_scaled[block_vars].values
        scores = pca.fit_transform(X_block)[:, 0]
        loadings = pca.components_[0]
        
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

    X_pcr = sm.add_constant(pcr_data[[f'PC_{b}' for b in blocks_list]])
    model_pcr = sm.OLS(y_res, X_pcr).fit()
    print(model_pcr.summary())

    pcr_ols_df = pd.DataFrame({
        'Variable': model_pcr.params.index,
        'Coefficient': model_pcr.params.values,
        'Std_Error': model_pcr.bse.values,
        't_value': model_pcr.tvalues.values,
        'p_value': model_pcr.pvalues.values
    })
    out_pcr_ols = os.path.join(OUTPUT_DIR, 'no_economic_pcr_ols.csv')
    pcr_ols_df.to_csv(out_pcr_ols, index=False)
    
    loadings_df = pd.DataFrame(loadings_list)
    out_pcr_loadings = os.path.join(OUTPUT_DIR, 'no_economic_pcr_loadings.csv')
    loadings_df.to_csv(out_pcr_loadings, index=False)
    print(f"  Saved PCR OLS to: {out_pcr_ols}")

    # -------------------------------------------------------------------------
    # STEP F: METHOD 2: PATH ANALYSIS (SEM VIA SEQUENTIAL OLS) ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- Path Analysis (SEM) on Residuals ---")
    model_path_hea = sm.OLS(pcr_data['PC_Health'], sm.add_constant(pcr_data[['PC_Research', 'PC_Governance']])).fit()
    model_path_pbr = sm.OLS(y_res, sm.add_constant(pcr_data[['PC_Research', 'PC_Health', 'PC_Governance']])).fit()

    paths = {
        'Res -> Health': model_path_hea.params['PC_Research'],
        'Gov -> Health': model_path_hea.params['PC_Governance'],
        
        'Res -> PBR (Direct)': model_path_pbr.params['PC_Research'],
        'Hea -> PBR': model_path_pbr.params['PC_Health'],
        'Gov -> PBR (Direct)': model_path_pbr.params['PC_Governance']
    }

    indirect_res_hea = paths['Res -> Health'] * paths['Hea -> PBR']
    total_res = paths['Res -> PBR (Direct)'] + indirect_res_hea

    indirect_gov_hea = paths['Gov -> Health'] * paths['Hea -> PBR']
    total_gov = paths['Gov -> PBR (Direct)'] + indirect_gov_hea

    path_effects = [
        {'Path': 'Research -> PBR (Direct)', 'Effect': paths['Res -> PBR (Direct)'], 'Type': 'Direct'},
        {'Path': 'Research -> Health -> PBR', 'Effect': indirect_res_hea, 'Type': 'Indirect'},
        {'Path': 'Research (Total Effect)', 'Effect': total_res, 'Type': 'Total'},
        
        {'Path': 'Health -> PBR (Direct)', 'Effect': paths['Hea -> PBR'], 'Type': 'Direct'},
        
        {'Path': 'Governance -> PBR (Direct)', 'Effect': paths['Gov -> PBR (Direct)'], 'Type': 'Direct'},
        {'Path': 'Governance -> Health -> PBR', 'Effect': indirect_gov_hea, 'Type': 'Indirect'},
        {'Path': 'Governance (Total Effect)', 'Effect': total_gov, 'Type': 'Total'}
    ]
    path_df = pd.DataFrame(path_effects)
    print(path_df.to_string(index=False))
    
    out_path = os.path.join(OUTPUT_DIR, 'no_economic_path_analysis.csv')
    path_df.to_csv(out_path, index=False)
    print(f"  Saved Path Analysis to: {out_path}")

    # -------------------------------------------------------------------------
    # STEP G: METHOD 3: RANDOM FOREST PERMUTATION IMPORTANCE ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- Random Forest Permutation Importance on Residuals ---")
    X_rf = data_scaled[all_vars].values
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_rf, y_res)
    
    result = permutation_importance(rf, X_rf, y_res, n_repeats=10, random_state=42)
    
    rf_df = pd.DataFrame({
        'Variable': all_vars,
        'Gini_Importance': rf.feature_importances_,
        'Permutation_Importance_Mean': result.importances_mean,
        'Permutation_Importance_Std': result.importances_std
    }).sort_values('Permutation_Importance_Mean', ascending=False)
    
    var_to_block = {}
    for b_name, b_vars in blocks.items():
        for v in b_vars:
            var_to_block[v] = b_name
    rf_df['Block'] = rf_df['Variable'].map(var_to_block)
    
    print(rf_df.to_string(index=False))
    
    rf_block = rf_df.groupby('Block')[['Gini_Importance', 'Permutation_Importance_Mean']].sum().reset_index()
    rf_block = rf_block.sort_values('Permutation_Importance_Mean', ascending=False)
    print("\nBlock-Level ML Importances:")
    print(rf_block.to_string(index=False))
    
    out_rf = os.path.join(OUTPUT_DIR, 'no_economic_random_forest.csv')
    rf_df.to_csv(out_rf, index=False)
    print(f"  Saved RF Importances to: {out_rf}")

    # -------------------------------------------------------------------------
    # STEP H: METHOD 4: ELASTIC NET CV REGRESSION ON RESIDUALS
    # -------------------------------------------------------------------------
    print("\n--- Elastic Net CV Regression on Residuals ---")
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10, random_state=42)
    en.fit(X_rf, y_res)
    
    en_df = pd.DataFrame({
        'Variable': all_vars,
        'Block': [var_to_block[v] for v in all_vars],
        'ElasticNet_Coefficient': en.coef_
    })
    en_df_nonzero = en_df[en_df['ElasticNet_Coefficient'] != 0].sort_values('ElasticNet_Coefficient', key=abs, ascending=False)
    print("Elastic Net Non-Zero Coefficients:")
    print(en_df_nonzero.to_string(index=False))
    
    out_en = os.path.join(OUTPUT_DIR, 'no_economic_elastic_net.csv')
    en_df.to_csv(out_en, index=False)
    print(f"  Saved Elastic Net Coefficients to: {out_en}")

    print("\n" + "="*80)
    print("NO ECONOMIC RESIDUAL ANALYSIS COMPLETE AND EXPORTED TO OUTPUT_DIR")
    print("="*80)

if __name__ == '__main__':
    main()
