"""
Figure 3: Five-Panel RQ2 Analysis
Disease vs Country Drivers of Clinical Trial Inequality

Panel A: Disease Contribution to Inequality (CIS with 95% CI)
Panel B: Lorenz Curves - Inequality Reduction by Removing Top 20% Diseases
Panel C: Temporal Inequality Reduction (2000-2024)
Panel D: Between-Disease vs Within-Disease Temporal Trends
Panel E: Country-Level Lorenz Curves - Inequality Reduction by Removing Top 20% Countries
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Color palettes
COLORS = {
    'primary_blue': '#2E86AB',
    'primary_red': '#A23B72',
    'accent_orange': '#F18F01',
    'danger_red': '#C73E1D',
    'info_blue': '#5E7CE2',
    'warning_orange': '#F4A261',
    'dark_green': '#264653',
    'light_yellow': '#E9C46A',
    'neutral_gray': '#6C757D',
    'light_gray': '#F8F9FA'
}

custom_diseases = [
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_gini_coefficient(values):
    """Calculate Gini coefficient"""
    if len(values) == 0:
        return 0
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def lorenz_curve_data(values):
    """Generate Lorenz curve coordinates"""
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])
    
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    total_sum = cumsum[-1]
    
    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0], cumsum / total_sum])
    
    return x, y


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_disease_ci_results():
    """Load Panel A data: Disease CIS with confidence intervals"""
    try:
        ci_results = pd.read_csv(r'c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data/rq2_bootstrap_confidence_intervals.csv')
        ci_results = ci_results.sort_values('CIS_Mean', ascending=True)
        print(f"[OK] Loaded disease CIS results: {len(ci_results)} diseases")
        return ci_results
    except:
        print("[WARN] Could not load rq2_bootstrap_confidence_intervals.csv")
        return None


def load_temporal_pbr_data():
    """Load temporal PBR data for all analyses"""
    try:
        # Load public aggregated dataset (70k for RQ2)
        base_dir = r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data"
        aggregated_file = os.path.join(base_dir, "public_aggregated_participants_70k.csv")
        gbd_file = os.path.join(base_dir, "gbddisease.csv")
        
        print(f"Loading data from {aggregated_file}...")
        try:
            temporal_pbr_data = pd.read_csv(aggregated_file)
        except FileNotFoundError:
            print(f"Error: File {aggregated_file} not found. Please run create_public_dataset.py first.")
            return None

        # Load GBD disease burden data for DALYs
        print(f"Loading data from {gbd_file}...")
        gbddisease = pd.read_csv(gbd_file)
        
        # Process GBD data to match aggregated format (Country-Disease-Year)
        # Note: We need disease mapping if we want to merge by name, but aggregated file has names.
        # We will assume temporal_pbr_data already has 'Disease' column with names.
        
        # To calculate PBR, we need DALYs. We need to map Disease Names in aggregated file to IDs in GBD, 
        # OR we need a way to get DALYs attached. 
        # Let's load mapping file just for Name -> ID resolution
        disease_mapping_file = os.path.join(base_dir, "disease_mapping.csv")
        disease_mapping = pd.read_csv(disease_mapping_file)
        
        # Merge DALYs into temporal_pbr_data
        # 1. Map Disease Name to ID
        temporal_pbr_data = temporal_pbr_data.merge(disease_mapping[['REI ID', 'REI Name']], 
                                                   left_on='Disease', right_on='REI Name', how='left')
        
        # 2. Merge GBD DALYs using ID, ISO3, Year
        # Group GBD by ISO3, cause_id, year (should be unique usually, or sum)
        gbd_agg = gbddisease.groupby(['ISO3', 'cause_id', 'year']).agg({'val': 'sum'}).reset_index()
        gbd_agg.rename(columns={'val': 'Avg_DALYs', 'year': 'YEAR'}, inplace=True)
        
        merged_data = temporal_pbr_data.merge(gbd_agg, 
                                             left_on=['ISO3', 'REI ID', 'YEAR'], 
                                             right_on=['ISO3', 'cause_id', 'YEAR'], 
                                             how='inner')
        
        # Filter to Level 2 diseases (already done by public_aggregated_participants_70k.csv)
        # Filter to custom_diseases (already done by public_aggregated_participants_70k.csv)
        
        # Rename columns to match expected format for PBR calculation
        merged_data.rename(columns={
            'Total_Participants': 'Total_Participants', # Already correct
            'Study_Count': 'Study_Count',               # Already correct
            'Disease': 'Disease'                        # Already correct
        }, inplace=True)

        # Ensure only custom diseases are included, though the aggregated file should already handle this
        merged_data = merged_data[merged_data['Disease'].isin(custom_diseases)]

        temporal_pbr_data = merged_data # Assign the merged data back to temporal_pbr_data

        # === 修改开始：使用你的调整公式计算PBR ===
        print("\nCalculating PBR using adjusted formula...")

        # Initialize PBR column as NaN
        temporal_pbr_data['PBR'] = np.nan

        # 按疾病和年份分组计算PBR
        for (disease, year), group in temporal_pbr_data.groupby(['Disease', 'YEAR']):
            # 只使用有正值的记录
            valid_mask = (group['Total_Participants'] > 0) & (group['Avg_DALYs'] > 0)
            valid_data = group[valid_mask].copy()

            if len(valid_data) >= 2:  # 至少需要两个国家
                # 计算全球总和
                total_participants = valid_data['Total_Participants'].sum()
                total_dalys = valid_data['Avg_DALYs'].sum()

                # 你的标准公式
                participant_shares = valid_data['Total_Participants'] / total_participants
                daly_shares = valid_data['Avg_DALYs'] / total_dalys

                # 应用你的调整
                min_daly_share = 0.001
                adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
                corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)

                # 更新到原数据框
                valid_indices = valid_data.index
                temporal_pbr_data.loc[valid_indices, 'PBR'] = corrected_pbr
                temporal_pbr_data.loc[valid_indices, 'Participant_Share'] = participant_shares
                temporal_pbr_data.loc[valid_indices, 'DALY_Share'] = daly_shares

        # 填充缺失值
        temporal_pbr_data['PBR'] = temporal_pbr_data['PBR'].fillna(0)
        temporal_pbr_data = temporal_pbr_data[~np.isinf(temporal_pbr_data['PBR'])]
        temporal_pbr_data = temporal_pbr_data[temporal_pbr_data['PBR'] > 0]
        # === 修改结束 ===

        print(f"[OK] Loaded temporal PBR data: {len(temporal_pbr_data)} records")
        print(f"   Years covered: {temporal_pbr_data['YEAR'].min()}-{temporal_pbr_data['YEAR'].max()}")
        print(f"   Unique diseases: {temporal_pbr_data['Disease'].nunique()}")
        print(f"   Unique countries: {temporal_pbr_data['ISO3'].nunique()}")
        print(f"   PBR range: [{temporal_pbr_data['PBR'].min():.6f}, {temporal_pbr_data['PBR'].max():.6f}]")
        print(f"   Mean PBR: {temporal_pbr_data['PBR'].mean():.6f}")

        return temporal_pbr_data

    except Exception as e:
        print(f"[WARN] Error loading temporal PBR data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_pbr_data():
    """Load aggregated PBR data for country-level analysis"""
    try:
        # Load from existing results if available
        disease_results = pd.read_csv(r'c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data/rq2_disease_inequality_results.csv')
        country_results = pd.read_csv(r'c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data/rq2_country_inequality_results.csv')

        # Reconstruct PBR data from disease results
        pbr_records = []
        for _, row in disease_results.iterrows():
            # Create approximate country-level records
            # This is a simplified reconstruction
            pbr_records.append({
                'Disease': row['Disease'],
                'Total_Participants': row['Total_Participants'],
                'Total_DALYs': row.get('Total_DALYs', 1),
                'Gini_Among_Participants': row['Gini_Among_Participants']
            })

        pbr_data = pd.DataFrame(pbr_records)
        print(f"[OK] Loaded PBR data: {len(pbr_data)} records")
        return pbr_data

    except Exception as e:
        print(f"[WARN]️ Error loading PBR data: {e}")
        return None


# ============================================================================
# PANEL B: LORENZ CURVES FOR DISEASE REMOVAL
# X-axis: Cumulative share of DISEASES (not countries)
# ============================================================================

def calculate_disease_lorenz_curves_by_disease(temporal_pbr_data, top_diseases):
    """
    Calculate Lorenz curves at DISEASE level
    X-axis: Cumulative share of diseases
    Y-axis: Cumulative share of participants

    IMPORTANT: This should match Panel C's calculation method!
    Both should use the SAME aggregation approach for consistency.
    """
    print("\n[GRAPH] Calculating disease-level Lorenz curves...")

    # Use COUNTRY-LEVEL aggregation like Panel C for consistency
    # Aggregate to country level, calculate Participants per DALY
    all_data = temporal_pbr_data.groupby(['ISO3', 'Disease']).agg({
        'Total_Participants': 'sum',
        'Avg_DALYs': 'sum'
    }).reset_index()

    # Calculate country totals across all diseases
    country_totals_all = all_data.groupby('ISO3').agg({
        'Total_Participants': 'sum',
        'Avg_DALYs': 'sum'
    })
    country_totals_all['Participants_per_DALY'] = (
        country_totals_all['Total_Participants'] / country_totals_all['Avg_DALYs']
    )
    all_participants_per_daly = country_totals_all['Participants_per_DALY'].values

    # Calculate without top 20% diseases
    reduced_data = all_data[~all_data['Disease'].isin(top_diseases)]
    country_totals_reduced = reduced_data.groupby('ISO3').agg({
        'Total_Participants': 'sum',
        'Avg_DALYs': 'sum'
    })
    country_totals_reduced['Participants_per_DALY'] = (
        country_totals_reduced['Total_Participants'] / country_totals_reduced['Avg_DALYs']
    )
    reduced_participants_per_daly = country_totals_reduced['Participants_per_DALY'].values

    # Generate Lorenz curves
    x_all, y_all = lorenz_curve_data(all_participants_per_daly)
    x_reduced, y_reduced = lorenz_curve_data(reduced_participants_per_daly)

    # Calculate Gini coefficients
    gini_all = calculate_gini_coefficient(all_participants_per_daly)
    gini_reduced = calculate_gini_coefficient(reduced_participants_per_daly)

    reduction_pct = ((gini_all - gini_reduced) / gini_all * 100) if gini_all > 0 else 0

    print(f"   Gini all diseases: {gini_all:.4f}")
    print(f"   Gini without top 20%: {gini_reduced:.4f}")
    print(f"   Reduction: {reduction_pct:.2f}%")

    return {
        'x_all': x_all, 'y_all': y_all, 'gini_all': gini_all,
        'x_reduced': x_reduced, 'y_reduced': y_reduced, 'gini_reduced': gini_reduced,
        'reduction_pct': reduction_pct
    }


# ============================================================================
# PANEL C: TEMPORAL INEQUALITY REDUCTION (LINE PLOT FORMAT)
# ============================================================================

def calculate_temporal_inequality_reduction(temporal_pbr_data, top_diseases):
    """
    Calculate Gini trends over time with and without top 20% diseases
    Format: Two line plots like in create_temporal_inequality_visualization
    """
    print("\n[GRAPH] Calculating temporal Gini trends...")

    temporal_pbr_data['Period'] = (temporal_pbr_data['YEAR'] // 2) * 2
    periods = sorted(temporal_pbr_data['Period'].unique())

    results = []
    for period in periods:
        period_data = temporal_pbr_data[temporal_pbr_data['Period'] == period]

        # Aggregate by Country-Disease
        period_agg = period_data.groupby(['ISO3', 'Disease']).agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'sum'
        }).reset_index()

        # Calculate country totals WITH all diseases
        country_totals_all = period_agg.groupby('ISO3').agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'sum'
        })
        country_totals_all['PBR'] = (
            country_totals_all['Total_Participants'] / country_totals_all['Avg_DALYs']
        )
        all_pbr = country_totals_all['PBR'].values
        all_pbr = all_pbr[~np.isinf(all_pbr)]
        all_pbr = all_pbr[all_pbr > 0]

        if len(all_pbr) < 10:
            continue

        gini_all = calculate_gini_coefficient(all_pbr)

        # Calculate country totals WITHOUT top 20% diseases
        period_reduced = period_agg[~period_agg['Disease'].isin(top_diseases)]
        if len(period_reduced) > 0:
            country_totals_reduced = period_reduced.groupby('ISO3').agg({
                'Total_Participants': 'sum',
                'Avg_DALYs': 'sum'
            })
            country_totals_reduced['PBR'] = (
                country_totals_reduced['Total_Participants'] / country_totals_reduced['Avg_DALYs']
            )
            reduced_pbr = country_totals_reduced['PBR'].values
            reduced_pbr = reduced_pbr[~np.isinf(reduced_pbr)]
            reduced_pbr = reduced_pbr[reduced_pbr > 0]

            gini_reduced = calculate_gini_coefficient(reduced_pbr)
        else:
            gini_reduced = gini_all

        results.append({
            'Period': period,
            'Gini_All': gini_all,
            'Gini_Reduced': gini_reduced,
            'Inequality_Reduction': ((gini_all - gini_reduced) / gini_all * 100) if gini_all > 0 else 0
        })

    results_df = pd.DataFrame(results)
    print(f"   Calculated for {len(results_df)} periods")
    return results_df


# ============================================================================
# PANEL D: BETWEEN VS WITHIN DISEASE TEMPORAL TRENDS
# ============================================================================

def calculate_theil_decomposition(period_agg, custom_diseases):
    """Calculate Theil decomposition for a single period"""
    disease_means = period_agg.groupby('Disease')['PBR'].agg(['mean', 'count']).reset_index()
    disease_means.columns = ['Disease', 'Disease_Mean_PBR', 'Disease_Count']

    overall_mean = period_agg['PBR'].mean()
    total_n = len(period_agg)

    # Between-disease component
    between_component = 0
    for _, row in disease_means.iterrows():
        n_g = row['Disease_Count']
        mean_g = row['Disease_Mean_PBR']

        if mean_g > 0 and overall_mean > 0:
            between_component += (n_g / total_n) * (mean_g / overall_mean) * np.log(mean_g / overall_mean)

    # Within-disease component
    within_component = 0
    for disease in disease_means['Disease']:
        disease_data = period_agg[period_agg['Disease'] == disease]
        mean_g = disease_data['PBR'].mean()

        for value in disease_data['PBR']:
            if value > 0 and mean_g > 0 and overall_mean > 0:
                within_component += (1 / total_n) * (value / overall_mean) * np.log(value / mean_g)

    total_theil = between_component + within_component

    if total_theil > 0:
        between_pct = (between_component / total_theil * 100)
        within_pct = (within_component / total_theil * 100)
    else:
        between_pct = within_pct = 0

    return {
        'total_theil': total_theil,
        'between_percent': between_pct,
        'within_percent': within_pct,
        'between_component': between_component,
        'within_component': within_component
    }


def calculate_temporal_theil_trends(temporal_pbr_data, custom_diseases, n_bootstrap=100):
    """Calculate temporal trends in Theil decomposition WITH bootstrap confidence intervals"""
    print("\n[GRAPH] Calculating temporal Theil decomposition with CIs...")

    temporal_pbr_data['Period'] = (temporal_pbr_data['YEAR'] // 2) * 2
    periods = sorted(temporal_pbr_data['Period'].unique())

    results = []
    bootstrap_results = []

    for period in periods:
        period_data = temporal_pbr_data[temporal_pbr_data['Period'] == period]

        period_agg = period_data.groupby(['ISO3', 'Disease']).agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'mean'
        }).reset_index()

        period_agg['PBR'] = period_agg['Total_Participants'] / period_agg['Avg_DALYs']
        period_agg = period_agg[~np.isinf(period_agg['PBR'])]
        period_agg = period_agg[period_agg['PBR'] > 0]

        if len(period_agg) < 30:
            continue

        theil_result = calculate_theil_decomposition(period_agg, custom_diseases)

        results.append({
            'Period': period,
            'Between_Percent': theil_result['between_percent'],
            'Within_Percent': theil_result['within_percent'],
            'Total_Theil': theil_result['total_theil']
        })

        # Bootstrap for confidence intervals
        print(f"   Bootstrapping {period}...")
        bootstrap_between = []
        bootstrap_within = []

        unique_countries = period_agg['ISO3'].unique()
        for boot_i in range(n_bootstrap):
            # Resample countries with replacement
            boot_countries = np.random.choice(unique_countries, size=len(unique_countries), replace=True)
            boot_data = []
            for country in boot_countries:
                country_data = period_agg[period_agg['ISO3'] == country]
                boot_data.append(country_data)

            if len(boot_data) > 0:
                boot_period_agg = pd.concat(boot_data, ignore_index=True)
                boot_theil = calculate_theil_decomposition(boot_period_agg, custom_diseases)

                if boot_theil is not None:
                    bootstrap_between.append(boot_theil['between_percent'])
                    bootstrap_within.append(boot_theil['within_percent'])

        # Calculate confidence intervals
        if len(bootstrap_between) > 10:
            bootstrap_results.append({
                'Period': period,
                'Between_CI_Lower': np.percentile(bootstrap_between, 2.5),
                'Between_CI_Upper': np.percentile(bootstrap_between, 97.5),
                'Within_CI_Lower': np.percentile(bootstrap_within, 2.5),
                'Within_CI_Upper': np.percentile(bootstrap_within, 97.5)
            })

    results_df = pd.DataFrame(results)
    bootstrap_df = pd.DataFrame(bootstrap_results)

    # Merge CIs
    if len(bootstrap_df) > 0:
        results_df = results_df.merge(bootstrap_df, on='Period', how='left')

    print(f"   Calculated for {len(results_df)} periods")
    return results_df


# ============================================================================
# PANEL E: COUNTRY-LEVEL LORENZ CURVES (NEW)
# ============================================================================

def identify_top_n_countries(temporal_pbr_data, n):
    """Identify top N countries by total participants"""
    country_totals = temporal_pbr_data.groupby('ISO3')['Total_Participants'].sum().sort_values(ascending=False)
    top_n_countries = country_totals.head(n).index.tolist()

    print(f"\n[WORLD] Top {n} countries by participants (removing {n}/{temporal_pbr_data['ISO3'].nunique()} = {n/temporal_pbr_data['ISO3'].nunique()*100:.1f}%):")
    for i, country in enumerate(top_n_countries, 1):
        print(f"   {i}. {country}: {country_totals[country]:,.0f} participants")

    return top_n_countries


def calculate_country_lorenz_curves(temporal_pbr_data, top_countries):
    """
    Calculate Lorenz curves at COUNTRY level (not country-disease pairs)
    X-axis: Cumulative share of COUNTRIES
    Y-axis: Cumulative share of participants
    """
    print("\n[GRAPH] Calculating country-level Lorenz curves...")

    # Aggregate to COUNTRY level (sum across all diseases) WITH all countries
    country_totals_all = temporal_pbr_data.groupby('ISO3').agg({
        'Total_Participants': 'sum'
    })
    all_participants = country_totals_all['Total_Participants'].values

    # WITHOUT top 20% countries
    country_totals_reduced = temporal_pbr_data[~temporal_pbr_data['ISO3'].isin(top_countries)].groupby('ISO3').agg({
        'Total_Participants': 'sum'
    })
    reduced_participants = country_totals_reduced['Total_Participants'].values

    # Generate Lorenz curves
    x_all, y_all = lorenz_curve_data(all_participants)
    x_reduced, y_reduced = lorenz_curve_data(reduced_participants)

    # Calculate Gini coefficients
    gini_all = calculate_gini_coefficient(all_participants)
    gini_reduced = calculate_gini_coefficient(reduced_participants)

    reduction_pct = ((gini_all - gini_reduced) / gini_all * 100) if gini_all > 0 else 0

    print(f"   Gini all countries: {gini_all:.4f}")
    print(f"   Gini without top 20%: {gini_reduced:.4f}")
    print(f"   Reduction: {reduction_pct:.2f}%")

    return {
        'x_all': x_all, 'y_all': y_all, 'gini_all': gini_all,
        'x_reduced': x_reduced, 'y_reduced': y_reduced, 'gini_reduced': gini_reduced,
        'reduction_pct': reduction_pct
    }


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_figure3_five_panels(include_panel_c=True):
    """Create comprehensive Figure 3 with 4 or 5 panels"""
    print("\n" + "="*80)
    if include_panel_c:
        print("CREATING FIGURE 3: FIVE-PANEL RQ2 ANALYSIS (SUPPLEMENTARY)")
    else:
        print("CREATING FIGURE 3: FOUR-PANEL RQ2 ANALYSIS (MAIN TEXT)")
    print("="*80)

    # Load data
    ci_results = load_disease_ci_results()
    temporal_pbr_data = load_temporal_pbr_data()

    if ci_results is None or temporal_pbr_data is None:
        print("❌ Required data files not found. Cannot create figure.")
        return None

    # Identify top diseases and countries
    ci_results_sorted = ci_results.sort_values('CIS_Mean', ascending=False)
    top_3_diseases = ci_results_sorted.head(3)['Disease'].tolist()

    print(f"\n[OK] Top 20% driver diseases (HIGHEST CIS = most inequality-driving):")
    for i, disease in enumerate(top_3_diseases, 1):
        cis = ci_results[ci_results['Disease'] == disease]['CIS_Mean'].values[0]
        print(f"   {i}. {disease}: CIS = {cis:.2f}%")

    print(f"\n[GRAPH] Bottom 3 diseases (NEGATIVE CIS = inequality-reducing/equalizing):")
    bottom_3 = ci_results_sorted.tail(3)['Disease'].tolist()
    for i, disease in enumerate(bottom_3, 1):
        cis = ci_results[ci_results['Disease'] == disease]['CIS_Mean'].values[0]
        print(f"   {i}. {disease}: CIS = {cis:.2f}%")

    # Calculate all panel data
    disease_lorenz = calculate_disease_lorenz_curves_by_disease(temporal_pbr_data, top_3_diseases)
    temporal_reduction = calculate_temporal_inequality_reduction(temporal_pbr_data, top_3_diseases)
    temporal_theil = calculate_temporal_theil_trends(temporal_pbr_data, custom_diseases)

    # For Panel E: Try removing top 20% of countries
    total_countries = temporal_pbr_data['ISO3'].nunique()
    n_countries_to_remove = max(3, int(total_countries * 0.20))
    top_n_countries = identify_top_n_countries(temporal_pbr_data, n_countries_to_remove)
    country_lorenz = calculate_country_lorenz_curves(temporal_pbr_data, top_n_countries)

    # Store results for statistical testing
    results_dict = {
        'disease_lorenz': disease_lorenz,
        'country_lorenz': country_lorenz,
        'temporal_reduction': temporal_reduction,
        'temporal_theil': temporal_theil,
        'n_countries_removed': n_countries_to_remove,
        'total_countries': total_countries
    }

    # Create figure with conditional layout
    if include_panel_c:
        # 5-panel layout: 3 panels top row, 2 panels bottom row centered
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
    else:
        # 4-panel layout: 2x2 grid
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ========================================================================
    # PANEL A: Disease Contribution to Inequality
    # ========================================================================
    if include_panel_c:
        ax1 = fig.add_subplot(gs[0, 0:2])  # Spans columns 0-1
    else:
        ax1 = fig.add_subplot(gs[0, 0])  # Top left

    y_pos = range(len(ci_results))
    bars = ax1.barh(y_pos, ci_results['CIS_Mean'], color='#CC6699', alpha=0.6)

    # Add error bars
    ax1.errorbar(ci_results['CIS_Mean'], y_pos,
                 xerr=[ci_results['CIS_Mean'] - ci_results['CIS_CI_Lower'],
                       ci_results['CIS_CI_Upper'] - ci_results['CIS_Mean']],
                 fmt='none', color='black', capsize=3, linewidth=1.5, alpha=0.7)

    # Add vertical line at x=0
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Contribution to Inequality Score (CIS) with 95% CI', fontsize=12, fontweight='bold')
    ax1.set_title('A. Disease Contribution to Global Inequality', fontsize=14, fontweight='bold', pad=15)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([d[:35] + '...' if len(d) > 35 else d for d in ci_results['Disease']], fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # ========================================================================
    # PANEL B: Disease Lorenz Curves
    # ========================================================================
    if include_panel_c:
        ax2 = fig.add_subplot(gs[0, 2:4])  # Spans columns 2-3
    else:
        ax2 = fig.add_subplot(gs[0, 1])  # Top right

    ax2.plot([0, 1], [0, 1], '--', color='black', alpha=0.5, linewidth=2, label='Perfect Equality')
    ax2.plot(disease_lorenz['x_all'], disease_lorenz['y_all'],
             color='#FF0099', linewidth=3,
             label=f"All Diseases (Gini={disease_lorenz['gini_all']:.3f})")
    ax2.plot(disease_lorenz['x_reduced'], disease_lorenz['y_reduced'],
             color='#0099CC', linewidth=3,
             label=f"Top 20% Removed (Gini={disease_lorenz['gini_reduced']:.3f})")

    ax2.fill_between(disease_lorenz['x_all'], disease_lorenz['y_all'],
                     [0] * len(disease_lorenz['x_all']), alpha=0.15, color='#FF0099')
    ax2.fill_between(disease_lorenz['x_reduced'], disease_lorenz['y_reduced'],
                     [0] * len(disease_lorenz['x_reduced']), alpha=0.15, color='#0099CC')

    ax2.set_xlabel('Cumulative Share of Countries', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Share of Participants', fontsize=12, fontweight='bold')
    ax2.set_title(f'B. Disease-Level Inequality Reduction',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ========================================================================
    # PANEL C: Temporal Gini Trends (ONLY IN 5-PANEL VERSION)
    # ========================================================================
    if include_panel_c:
        ax3 = fig.add_subplot(gs[0, 4:6])  # Spans columns 4-5

        # Plot with smoothing using scipy's UnivariateSpline
        from scipy.interpolate import UnivariateSpline

        periods = temporal_reduction['Period'].values
        gini_all = temporal_reduction['Gini_All'].values
        gini_reduced = temporal_reduction['Gini_Reduced'].values

        # Create smooth curves with INCREASED smoothing
        if len(periods) >= 4:
            period_smooth = np.linspace(periods.min(), periods.max(), 100)

            try:
                # Smoothing parameter: INCREASED from 0.001 to 0.02 for less shaky curves
                spline_all = UnivariateSpline(periods, gini_all, s=len(periods) * 0.02, k=3)
                spline_reduced = UnivariateSpline(periods, gini_reduced, s=len(periods) * 0.02, k=3)

                smooth_all = spline_all(period_smooth)
                smooth_reduced = spline_reduced(period_smooth)

                # Plot smooth lines
                ax3.plot(period_smooth, smooth_all, '-',
                         color=COLORS['primary_blue'], linewidth=3, alpha=0.8, label='All Diseases')
                ax3.plot(period_smooth, smooth_reduced, '-',
                         color=COLORS['primary_red'], linewidth=3, alpha=0.8, label='Top 20% Removed')

                # Add scatter points for observed data
                ax3.scatter(periods, gini_all, color=COLORS['primary_blue'],
                           s=60, alpha=0.6, zorder=3, edgecolors='white', linewidth=1)
                ax3.scatter(periods, gini_reduced, color=COLORS['primary_red'],
                           s=60, alpha=0.6, zorder=3, edgecolors='white', linewidth=1, marker='s')
            except:
                # Fallback to regular plotting if smoothing fails
                ax3.plot(periods, gini_all, 'o-',
                         color=COLORS['primary_blue'], linewidth=2.5, markersize=8, label='All Diseases')
                ax3.plot(periods, gini_reduced, 's-',
                         color=COLORS['primary_red'], linewidth=2.5, markersize=8, label='Top 20% Removed')
        else:
            ax3.plot(periods, gini_all, 'o-',
                     color=COLORS['primary_blue'], linewidth=2.5, markersize=8, label='All Diseases')
            ax3.plot(periods, gini_reduced, 's-',
                     color=COLORS['primary_red'], linewidth=2.5, markersize=8, label='Top 20% Removed')

        # Add trend lines
        if len(temporal_reduction) > 2:
            slope_all, intercept_all, r_all, p_all, _ = stats.linregress(
                temporal_reduction['Period'], temporal_reduction['Gini_All'])
            trend_all = slope_all * temporal_reduction['Period'] + intercept_all
            ax3.plot(temporal_reduction['Period'], trend_all, '--',
                    color=COLORS['primary_blue'], linewidth=2, alpha=0.5)

            slope_red, intercept_red, r_red, p_red, _ = stats.linregress(
                temporal_reduction['Period'], temporal_reduction['Gini_Reduced'])
            trend_red = slope_red * temporal_reduction['Period'] + intercept_red
            ax3.plot(temporal_reduction['Period'], trend_red, '--',
                    color=COLORS['primary_red'], linewidth=2, alpha=0.5)

            # Add statistics text
            ax3.text(0.05, 0.95,
                    f'All: slope={slope_all:.4f}, R²={r_all**2:.3f}\nTop 20% removed: slope={slope_red:.4f}, R²={r_red**2:.3f}',
                    transform=ax3.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')

        ax3.set_xlabel('Period (2-year bins)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
        ax3.set_title('C. Temporal Inequality Evolution\n(Removing Top 20% Diseases)',
                      fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(2000, 2026, 4))
        ax3.set_ylim(0, 1.0)

    # ========================================================================
    # PANEL D (or C in 4-panel): Between vs Within Disease Trends
    # ========================================================================
    if include_panel_c:
        ax4 = fig.add_subplot(gs[1, 1:3])  # Centered: columns 1-2
        panel_label = "(D)"
    else:
        ax4 = fig.add_subplot(gs[1, 0])  # Bottom left
        panel_label = "(C)"

    from scipy.interpolate import UnivariateSpline

    periods = temporal_theil['Period'].values
    between_pct = temporal_theil['Between_Percent'].values
    within_pct = temporal_theil['Within_Percent'].values

    # Create smooth curves for visualization
    if len(periods) >= 4:
        period_smooth = np.linspace(periods.min(), periods.max(), 100)

        try:
            # Smooth CI bands if available
            if 'Between_CI_Lower' in temporal_theil.columns:
                between_ci_lower = temporal_theil['Between_CI_Lower'].values
                between_ci_upper = temporal_theil['Between_CI_Upper'].values

                spline_ci_lower = UnivariateSpline(periods, between_ci_lower, s=len(periods) * 1.5, k=3)
                spline_ci_upper = UnivariateSpline(periods, between_ci_upper, s=len(periods) * 1.5, k=3)
                smooth_ci_lower = spline_ci_lower(period_smooth)
                smooth_ci_upper = spline_ci_upper(period_smooth)

                ax4.fill_between(period_smooth, smooth_ci_lower, smooth_ci_upper,
                                color='#CC6699', alpha=0.4, label='Between 95% CI')

            if 'Within_CI_Lower' in temporal_theil.columns:
                within_ci_lower = temporal_theil['Within_CI_Lower'].values
                within_ci_upper = temporal_theil['Within_CI_Upper'].values

                spline_within_ci_lower = UnivariateSpline(periods, within_ci_lower, s=len(periods) * 1.5, k=3)
                spline_within_ci_upper = UnivariateSpline(periods, within_ci_upper, s=len(periods) * 1.5, k=3)
                smooth_within_ci_lower = spline_within_ci_lower(period_smooth)
                smooth_within_ci_upper = spline_within_ci_upper(period_smooth)

                ax4.fill_between(period_smooth, smooth_within_ci_lower, smooth_within_ci_upper,
                                color='#CC9966', alpha=0.4, label='Within 95% CI')

            # Smooth trend lines for observed data
            spline_between = UnivariateSpline(periods, between_pct, s=len(periods) * 2, k=3)
            spline_within = UnivariateSpline(periods, within_pct, s=len(periods) * 2, k=3)

            smooth_between = spline_between(period_smooth)
            smooth_within = spline_within(period_smooth)

            ax4.plot(period_smooth, smooth_between, '-', color='#CC6699',
                    linewidth=3, alpha=0.8, label='Between-Disease (smooth)')
            ax4.plot(period_smooth, smooth_within, '-', color='#CC9966',
                    linewidth=3, alpha=0.8, label='Within-Disease (smooth)')
        except:
            # Fallback to regular CI bands
            if 'Between_CI_Lower' in temporal_theil.columns:
                ax4.fill_between(periods,
                                temporal_theil['Between_CI_Lower'],
                                temporal_theil['Between_CI_Upper'],
                                color='#CC6699', alpha=0.2, label='Between 95% CI')

            if 'Within_CI_Lower' in temporal_theil.columns:
                ax4.fill_between(periods,
                                temporal_theil['Within_CI_Lower'],
                                temporal_theil['Within_CI_Upper'],
                                color='#CC9966', alpha=0.2, label='Within 95% CI')

    # Plot scatter points for observed data
    ax4.scatter(periods, between_pct, color='#CC6699',
                s=80, alpha=0.7, zorder=3, edgecolors='white', linewidth=1)
    ax4.scatter(periods, within_pct, color='#CC9966',
                s=80, alpha=0.7, zorder=3, edgecolors='white', linewidth=1)

    # Linear trend lines (DASHED)
    slope_b, intercept_b, r_value_b, p_value_b, _ = stats.linregress(periods, between_pct)
    trend_line_b = slope_b * periods + intercept_b
    ax4.plot(periods, trend_line_b, '--', color='#CC6699',
             linewidth=2, alpha=0.6)

    slope_w, intercept_w, r_value_w, p_value_w, _ = stats.linregress(periods, within_pct)
    trend_line_w = slope_w * periods + intercept_w
    ax4.plot(periods, trend_line_w, '--', color='#CC9966',
             linewidth=2, alpha=0.6)

    ax4.set_xlabel('Period Start Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% of Total Inequality', fontsize=12, fontweight='bold')
    ax4.set_title(f'C. Theil Decomposition Temporal Trends', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(range(2000, 2026, 4))
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=9, loc='center right')
    ax4.grid(True, alpha=0.3)

    # Add statistical annotation
    ax4.text(0.02, 0.98,
             f"Between: {slope_b:.2f}%/period (R²={r_value_b**2:.3f}, p={p_value_b:.3f})\nWithin: {slope_w:.2f}%/period (R²={r_value_w**2:.3f}, p={p_value_w:.3f})",
             transform=ax4.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # ========================================================================
    # PANEL E (or D in 4-panel): Country Lorenz Curves
    # ========================================================================
    if include_panel_c:
        ax5 = fig.add_subplot(gs[1, 3:5])  # Centered: columns 3-4
        panel_label = "(E)"
    else:
        ax5 = fig.add_subplot(gs[1, 1])  # Bottom right
        panel_label = "(D)"

    ax5.plot([0, 1], [0, 1], '--', color='black', alpha=0.5, linewidth=2, label='Perfect Equality')
    ax5.plot(country_lorenz['x_all'], country_lorenz['y_all'],
             color='#FF0099', linewidth=3,
             label=f"All Countries (Gini={country_lorenz['gini_all']:.3f})")
    ax5.plot(country_lorenz['x_reduced'], country_lorenz['y_reduced'],
             color='#0099CC', linewidth=3,
             label=f"Top 20% Removed (Gini={country_lorenz['gini_reduced']:.3f})")

    ax5.fill_between(country_lorenz['x_all'], country_lorenz['y_all'],
                     [0] * len(country_lorenz['x_all']), alpha=0.15, color='#FF0099')
    ax5.fill_between(country_lorenz['x_reduced'], country_lorenz['y_reduced'],
                     [0] * len(country_lorenz['x_reduced']), alpha=0.15, color='#0099CC')

    ax5.set_xlabel('Cumulative Share of Countries', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Share of Participants', fontsize=12, fontweight='bold')
    ax5.set_title(f'D. Country-Level Inequality Reduction',
                  fontsize=14, fontweight='bold', pad=15)
    ax5.legend(fontsize=11, loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # ========================================================================
    # Save figure
    # ========================================================================
    plt.tight_layout()

    # Save with appropriate filename
    if include_panel_c:
        output_file = 'figure_3_rq2_five_panels_supplementary.png'
    else:
        output_file = 'figure_3_rq2_four_panels_main.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Figure saved: {output_file}")

    # Save as PDF (vector format)
    plt.savefig('figure_3_rq2_four_panels_main.pdf', bbox_inches='tight', facecolor='white')
    print(f"[OK] PDF saved.")

    # Save supplementary data (only once)
    if include_panel_c:
        temporal_reduction.to_csv('figure3_panel_c_temporal_reduction.csv', index=False)
        temporal_theil.to_csv('figure3_panel_d_theil_decomposition.csv', index=False)
        print("[OK] Supplementary data saved")

    plt.show()

    return fig, results_dict


# ============================================================================
# SUPPLEMENTARY ANALYSIS: STATISTICAL TESTING
# ============================================================================

def run_statistical_tests(temporal_reduction, temporal_theil, disease_lorenz, country_lorenz, n_countries_removed, total_countries):
    """Run statistical tests for temporal trends"""
    print("\n" + "="*80)
    print("STATISTICAL TESTING FOR FIGURE 3")
    print("="*80)

    # Test 1: Is temporal reduction trend significant?
    if len(temporal_reduction) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            temporal_reduction['Period'], temporal_reduction['Inequality_Reduction'])

        print("\n[GRAPH] Panel C: Temporal Inequality Reduction Trend")
        print(f"   Slope: {slope:.4f}% per period")
        print(f"   R^2: {r_value**2:.4f}")
        print(f"   p-value: {p_value:.4f}")
        print(f"   Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} trend")

    # Test 2: Are between/within trends diverging?
    if len(temporal_theil) > 2:
        slope_b, _, r_b, p_b, _ = stats.linregress(
            temporal_theil['Period'], temporal_theil['Between_Percent'])
        slope_w, _, r_w, p_w, _ = stats.linregress(
            temporal_theil['Period'], temporal_theil['Within_Percent'])

        print("\n[GRAPH] Panel D: Theil Decomposition Trends")
        print(f"   Between-disease slope: {slope_b:.4f}% per period (p={p_b:.4f})")
        print(f"   Within-disease slope: {slope_w:.4f}% per period (p={p_w:.4f})")
        print(f"   Trend difference: {abs(slope_b - slope_w):.4f}% per period")

        if p_b < 0.05 and p_w < 0.05:
            if slope_b < 0 and slope_w > 0:
                print("   [OK] CONFIRMED: Between-disease declining, Within-disease increasing")
                print("   → Country factors becoming MORE dominant over time")
            elif slope_b > 0 and slope_w < 0:
                print("   [WARN] UNEXPECTED: Between-disease increasing, Within-disease declining")
            else:
                print("   → Both trends moving in same direction")
        else:
            print("   [WARN]️  At least one trend not statistically significant")

    # Test 2: Are between/within trends diverging?
    if len(temporal_theil) > 2:
        slope_b, _, r_b, p_b, _ = stats.linregress(
            temporal_theil['Period'], temporal_theil['Between_Percent'])
        slope_w, _, r_w, p_w, _ = stats.linregress(
            temporal_theil['Period'], temporal_theil['Within_Percent'])

        print("\n[GRAPH] Panel D: Theil Decomposition Trends")
        print(f"   Between-disease slope: {slope_b:.4f}% per period (p={p_b:.4f})")
        print(f"   Within-disease slope: {slope_w:.4f}% per period (p={p_w:.4f})")
        print(f"   Trend difference: {abs(slope_b - slope_w):.4f}% per period")

        if p_b < 0.05 and p_w < 0.05:
            if slope_b < 0 and slope_w > 0:
                print("   [OK] CONFIRMED: Between-disease declining, Within-disease increasing")
                print("   → Country factors becoming MORE dominant over time")
            elif slope_b > 0 and slope_w < 0:
                print("   [WARN] UNEXPECTED: Between-disease increasing, Within-disease declining")
            else:
                print("   → Both trends moving in same direction")
        else:
            print("   [WARN]️  At least one trend not statistically significant")

    # Test 3: Gini consistency between panels
    print("\n[GRAPH] Panel B vs Panel C: Gini Consistency Check")
    print(f"   Panel B Gini (all diseases): {disease_lorenz['gini_all']:.4f}")
    print(f"   Panel C Gini range: {temporal_reduction['Gini_All'].min():.4f} - {temporal_reduction['Gini_All'].max():.4f}")
    print(f"   Panel C mean Gini: {temporal_reduction['Gini_All'].mean():.4f}")

    gini_diff = abs(disease_lorenz['gini_all'] - temporal_reduction['Gini_All'].mean())
    if gini_diff < 0.1:
        print(f"   [OK] CONSISTENT: Gini values are similar (difference: {gini_diff:.4f})")
    else:
        print(f"   [WARN] WARNING: Large difference between Panel B and Panel C Gini (difference: {gini_diff:.4f})")
        print(f"   → This suggests different aggregation methods are being used")
        print(f"   → Panel B should use the same method as Panel C for consistency")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FIGURE 3 GENERATION: RQ2 FIVE-PANEL ANALYSIS")
    print("Disease vs Country Drivers of Clinical Trial Inequality")
    print("="*80)

    # Create MAIN figure (4 panels, without Panel C)
    print("\n[INFO] Creating MAIN figure (4 panels)...")
    result_main = create_figure3_five_panels(include_panel_c=False)

    # Create SUPPLEMENTARY figure (5 panels, with Panel C)
    print("\n[GRAPH] Creating SUPPLEMENTARY figure (5 panels)...")
    result_supp = create_figure3_five_panels(include_panel_c=True)

    if result_main is not None and result_supp is not None:
        fig_main, results_dict = result_main

        # Run comprehensive statistical testing
        try:
            run_statistical_tests(
                results_dict['temporal_reduction'],
                results_dict['temporal_theil'],
                results_dict['disease_lorenz'],
                results_dict['country_lorenz'],
                results_dict['n_countries_removed'],
                results_dict['total_countries']
            )
        except Exception as e:
            print(f"\n[WARN] Could not run complete statistical tests: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)
        print("[OK] FIGURE 3 GENERATION COMPLETE")
        print("="*80)
        print("\nOutputs:")
        print("  [GRAPH] figure_3_rq2_four_panels_main.png (for main text)")
        print("  [GRAPH] figure_3_rq2_five_panels_supplementary.png (for SI)")
        print("  - figure3_panel_c_temporal_reduction.csv (Panel C data)")
        print("  - figure3_panel_d_theil_decomposition.csv (Panel D data)")
        print("\nFigure Structure:")
        print("  Panel A: Disease CIS with 95% CI (bootstrap)")
        print("  Panel B: Lorenz curves - disease removal effect")
        print("  Panel C: Temporal Gini trends (2000-2024)")
        print("  Panel D: Between/Within disease temporal trends WITH 95% CI shading")
        print("  Panel E: Lorenz curves - country removal effect")
        print("\nLayout:")
        print("  Row 1: Panels A, B, C (full width)")
        print("  Row 2: Panels D, E (centered)")
        print("\nKey Message:")
        print("  → Country factors (geography) dominate over disease factors")
        print("  → Disease removal has minimal effect on inequality")
        print("  → Country factors are INCREASING in importance over time")
        print("  → Between-disease inequality declining, within-disease increasing")
    else:
        print("\n❌ Figure generation failed. Check data files.")
        print("\nRequired files:")
        print("  - rq2_bootstrap_confidence_intervals.csv")
        print("  - year_70k.csv")
        print("  - pmid_cause_70k.csv")
        print("  - geoinfor.csv")
        print("  - gbddisease.csv")
        
    print("\n" + "="*80)