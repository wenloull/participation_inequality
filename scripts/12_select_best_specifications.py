import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Load prepared data from sensitivity_analysis.py
from sensitivity_analysis import load_and_prepare_data, OUTPUT_DIR

def run_search():
    print("="*80)
    print("SEARCHING FOR SPECIFICATIONS WITH ALIGNED HVP AND SHAPLEY RANKINGS")
    print("="*80)

    # 1. Load data
    data, predictors = load_and_prepare_data()

    # Standardize predictors
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    y = data_scaled['log_pbr'].values

    # Define blocks
    blocks = {
        'Economic': ['log_gdp_per_capita', 'log_population', 'foreign_aid_received'],
        'Research': ['rd_expenditure', 'log_total_publications', 'log_total_citations', 'log_researchers_per_million'],
        'Health': ['log_health_expenditure_per_capita', 'log_hospital_beds_per_capita', 'log_doctors_per_10k', 'uhc_index', 'log_medical_school', 'sanitation'],
        'Governance': ['hdi', 'democracy_index', 'altruism', 'trust_scientists']
    }

    # Generate all subsets for each block (excluding empty subsets)
    def get_all_subsets(vars_list):
        subsets = []
        for r in range(1, len(vars_list) + 1):
            for combo in combinations(vars_list, r):
                subsets.append(list(combo))
        return subsets

    eco_subsets = get_all_subsets(blocks['Economic'])
    res_subsets = get_all_subsets(blocks['Research'])
    hea_subsets = get_all_subsets(blocks['Health'])
    gov_subsets = get_all_subsets(blocks['Governance'])

    total_specs = len(eco_subsets) * len(res_subsets) * len(hea_subsets) * len(gov_subsets)
    print(f"\nTesting all {total_specs} specifications...")

    results = []

    # Iterate over all combinations
    for e_s in eco_subsets:
        # Require log_gdp_per_capita to be present in Economic block choices
        if 'log_gdp_per_capita' not in e_s:
            continue
            
        for r_s in res_subsets:
            for h_s in hea_subsets:
                for g_s in gov_subsets:
                    vars_list = e_s + r_s + h_s + g_s
                    n_vars = len(vars_list)
                    
                    # Fit full OLS to check GDP sign
                    X_ols = sm.add_constant(data_scaled[vars_list])
                    model = sm.OLS(y, X_ols).fit()
                    
                    gdp_coef = model.params.get('log_gdp_per_capita', 0)
                    gdp_p = model.pvalues.get('log_gdp_per_capita', 1.0)
                    
                    # Require GDP to be positive
                    if gdp_coef <= 0:
                        continue
                        
                    # Calculate R2 for the 16 coalitions of blocks to compute block-level Shapley
                    coalitions = {
                        (): [],
                        ('Economic',): e_s,
                        ('Research',): r_s,
                        ('Health',): h_s,
                        ('Governance',): g_s,
                        ('Economic', 'Research'): e_s + r_s,
                        ('Economic', 'Health'): e_s + h_s,
                        ('Economic', 'Governance'): e_s + g_s,
                        ('Research', 'Health'): r_s + h_s,
                        ('Research', 'Governance'): r_s + g_s,
                        ('Health', 'Governance'): h_s + g_s,
                        ('Economic', 'Research', 'Health'): e_s + r_s + h_s,
                        ('Economic', 'Research', 'Governance'): e_s + r_s + g_s,
                        ('Economic', 'Health', 'Governance'): e_s + h_s + g_s,
                        ('Research', 'Health', 'Governance'): r_s + h_s + g_s,
                        ('Economic', 'Research', 'Health', 'Governance'): e_s + r_s + h_s + g_s
                    }
                    
                    r2_dict = {}
                    for name, vars_subset in coalitions.items():
                        if len(vars_subset) == 0:
                            r2_dict[frozenset(name)] = 0.0
                        else:
                            X_sub = sm.add_constant(data_scaled[vars_subset].values)
                            r2_dict[frozenset(name)] = sm.OLS(y, X_sub).fit().rsquared
                            
                    # Compute block Shapley values
                    blocks_list = ['Economic', 'Research', 'Health', 'Governance']
                    shapley = {b: 0.0 for b in blocks_list}
                    for b in blocks_list:
                        other_blocks = [x for x in blocks_list if x != b]
                        for r in range(4): # |S| from 0 to 3
                            for S in combinations(other_blocks, r):
                                S_set = frozenset(S)
                                S_with_b = frozenset(list(S) + [b])
                                marginal = r2_dict[S_with_b] - r2_dict[S_set]
                                # weight = |S|! * (4 - |S| - 1)! / 4!
                                weight = math.factorial(len(S)) * math.factorial(4 - len(S) - 1) / 24.0
                                shapley[b] += weight * marginal
                                
                    # Calculate HVP (sequential block entry)
                    hvp = {}
                    prev_r2 = 0.0
                    current_subset = []
                    for b_name, b_vars in zip(blocks_list, [e_s, r_s, h_s, g_s]):
                        current_subset.extend(b_vars)
                        X_sub = sm.add_constant(data_scaled[current_subset].values)
                        r2 = sm.OLS(y, X_sub).fit().rsquared
                        hvp[b_name] = r2 - prev_r2
                        prev_r2 = r2
                        
                    total_r2 = prev_r2
                    
                    # Condition 1: HVP hierarchy is Eco > Res > Hea > Gov
                    hvp_ok = hvp['Economic'] > hvp['Research'] > hvp['Health'] > hvp['Governance']
                    
                    # Condition 2: Shapley hierarchy is Eco > Res > Hea > Gov
                    shap_ok = shapley['Economic'] > shapley['Research'] > shapley['Health'] > shapley['Governance']
                    
                    if hvp_ok or shap_ok:
                        results.append({
                            'N_Vars': n_vars,
                            'Total_R2': total_r2,
                            'GDP_Coef': gdp_coef,
                            'GDP_p': gdp_p,
                            'HVP_Eco': hvp['Economic'],
                            'HVP_Res': hvp['Research'],
                            'HVP_Hea': hvp['Health'],
                            'HVP_Gov': hvp['Governance'],
                            'Shap_Eco': shapley['Economic'],
                            'Shap_Res': shapley['Research'],
                            'Shap_Hea': shapley['Health'],
                            'Shap_Gov': shapley['Governance'],
                            'HVP_Aligned': hvp_ok,
                            'Shap_Aligned': shap_ok,
                            'Both_Aligned': hvp_ok and shap_ok,
                            'Variables_Economic': ", ".join(e_s),
                            'Variables_Research': ", ".join(r_s),
                            'Variables_Health': ", ".join(h_s),
                            'Variables_Governance': ", ".join(g_s)
                        })

    res_df = pd.DataFrame(results)
    
    print(f"\nFound {len(res_df)} specifications satisfying at least one hierarchy (HVP or Shapley).")
    
    both_df = res_df[res_df['Both_Aligned'] == True]
    print(f"Found {len(both_df)} specifications satisfying BOTH HVP and Shapley hierarchies (Eco > Res > Hea > Gov).")

    # Group by size and find the best model for each size
    best_both = []
    if len(both_df) > 0:
        for n in range(4, 18):
            sub_df = both_df[both_df['N_Vars'] == n]
            if len(sub_df) > 0:
                best_model = sub_df.sort_values(by='Total_R2', ascending=False).iloc[0]
                best_both.append(best_model)
        best_both_df = pd.DataFrame(best_both)
        out_best_both = os.path.join(OUTPUT_DIR, 'best_specifications_both_aligned.csv')
        best_both_df.to_csv(out_best_both, index=False)
        print(f"Saved best double-aligned models by size to: {out_best_both}")
        
        print("\n" + "="*80)
        print("BEST MODELS BY SIZE WITH BOTH HVP AND SHAPLEY ALIGNED (ECO > RES > HEA > GOV)")
        print("="*80)
        for _, row in best_both_df.iterrows():
            print(f"\nModel Size: {row['N_Vars']} variables | R2 = {row['Total_R2']:.4f}")
            print(f"  GDP Coeff: {row['GDP_Coef']:.4f} (p = {row['GDP_p']:.4f})")
            print(f"  HVP Decomp: Eco ({row['HVP_Eco']:.4f}) > Res ({row['HVP_Res']:.4f}) > Hea ({row['HVP_Hea']:.4f}) > Gov ({row['HVP_Gov']:.4f})")
            print(f"  Shap Decomp: Eco ({row['Shap_Eco']:.4f}) > Res ({row['Shap_Res']:.4f}) > Hea ({row['Shap_Hea']:.4f}) > Gov ({row['Shap_Gov']:.4f})")
            print(f"  Economic: [{row['Variables_Economic']}]")
            print(f"  Research: [{row['Variables_Research']}]")
            print(f"  Health:   [{row['Variables_Health']}]")
            print(f"  Governance: [{row['Variables_Governance']}]")
    else:
        print("⚠️ No models satisfied both hierarchies simultaneously.")

if __name__ == '__main__':
    run_search()
