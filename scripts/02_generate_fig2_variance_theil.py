"""
Figure 2: Decomposition Analysis of Participation Inequality (2000-2024)
Disease vs Country Drivers of Clinical Trial Inequality
Using new geoinfor_195k.csv data
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set style for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Color palette
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

# The 16 Level 2 diseases (excluding Enteric infections)
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
    'Substance use disorders'
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
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_temporal_pbr_data():
    """Load and prepare temporal PBR data from the 195k dataset"""
    print("\nLoading and preparing datasets...")
    
    pmid_cause_path = "/Users/wen/Desktop/participation_inequality/CauseClassier/pmid_cause.csv"
    geoinfor_path = "/Users/wen/Desktop/participation_inequality/analysiswoold/geoinfor195kwoold.csv"
    year_path = "/Users/wen/Desktop/participation_inequality/data/year_195k.csv"
    gbd_path = "/Users/wen/Desktop/participation_inequality/data/gbddisease.csv"

    # Read the data
    df_geo = pd.read_csv(geoinfor_path).rename(columns={'pmid': 'PMID', 'iso3': 'ISO3', 'amount': 'Amount'})
    df_year = pd.read_csv(year_path).rename(columns={'PMID': 'PMID', 'YEAR': 'YEAR'})
    df_cause = pd.read_csv(pmid_cause_path).rename(columns={'CAUSE': 'Cause'})
    df_gbd = pd.read_csv(gbd_path)

    # Filter year to 2000-2024
    df_year_filtered = df_year[(df_year['YEAR'] >= 2000) & (df_year['YEAR'] <= 2024)].copy()

    # Filter causes to Level 2 and custom diseases (16 diseases)
    df_cause_filtered = df_cause[
        (df_cause['Level'] == 2) &
        (df_cause['Cause'].isin(custom_diseases))
    ][['PMID', 'Cause', 'cause_id']].drop_duplicates().copy()

    # Merge trial data
    merged_trials = df_geo.merge(df_year_filtered, on='PMID', how='inner')
    merged_trials = merged_trials.merge(df_cause_filtered[['PMID', 'Cause', 'cause_id']], on='PMID', how='inner')

    # Aggregate by country-disease-year
    # Sum Amount as Total_Participants, count unique PMID as Study_Count
    aggregated_trials = merged_trials.groupby(['ISO3', 'Cause', 'cause_id', 'YEAR']).agg({
        'Amount': 'sum',
        'PMID': 'nunique'
    }).reset_index().rename(columns={
        'Cause': 'Disease',
        'Amount': 'Total_Participants',
        'PMID': 'Study_Count'
    })

    # Prepare GBD DALY data
    gbd_agg = df_gbd.groupby(['ISO3', 'cause_id', 'year']).agg({'val': 'sum'}).reset_index()
    gbd_agg.rename(columns={'val': 'Avg_DALYs', 'year': 'YEAR'}, inplace=True)

    # Merge GBD DALYs into aggregated_trials
    temporal_pbr_data = aggregated_trials.merge(gbd_agg, on=['ISO3', 'cause_id', 'YEAR'], how='inner')

    # Filter to unified 180 countries matching strict intersection
    unified_countries = pd.read_csv("/Users/wen/Desktop/participation_inequality/analysiswoold/unified_180_countries.csv")['ISO3'].unique()
    temporal_pbr_data = temporal_pbr_data[temporal_pbr_data['ISO3'].isin(unified_countries)]

    print("Calculating PBR using adjusted formula...")
    temporal_pbr_data['PBR'] = np.nan
    temporal_pbr_data['Participant_Share'] = np.nan
    temporal_pbr_data['DALY_Share'] = np.nan

    # Group by Disease and YEAR to calculate PBR
    for (disease, year), group in temporal_pbr_data.groupby(['Disease', 'YEAR']):
        valid_mask = (group['Total_Participants'] > 0) & (group['Avg_DALYs'] > 0)
        valid_data = group[valid_mask].copy()

        if len(valid_data) >= 2:
            total_participants = valid_data['Total_Participants'].sum()
            total_dalys = valid_data['Avg_DALYs'].sum()

            participant_shares = valid_data['Total_Participants'] / total_participants
            daly_shares = valid_data['Avg_DALYs'] / total_dalys

            min_daly_share = 0.001
            adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
            corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)

            valid_indices = valid_data.index
            temporal_pbr_data.loc[valid_indices, 'PBR'] = corrected_pbr
            temporal_pbr_data.loc[valid_indices, 'Participant_Share'] = participant_shares
            temporal_pbr_data.loc[valid_indices, 'DALY_Share'] = daly_shares

    # Clean PBR data
    temporal_pbr_data['PBR'] = temporal_pbr_data['PBR'].fillna(0)
    temporal_pbr_data = temporal_pbr_data[~np.isinf(temporal_pbr_data['PBR'])]
    temporal_pbr_data = temporal_pbr_data[temporal_pbr_data['PBR'] > 0]

    print(f"[OK] Loaded temporal PBR data: {len(temporal_pbr_data)} records")
    print(f"   Years covered: {temporal_pbr_data['YEAR'].min()}-{temporal_pbr_data['YEAR'].max()}")
    print(f"   Unique diseases: {temporal_pbr_data['Disease'].nunique()}")
    print(f"   Unique countries: {temporal_pbr_data['ISO3'].nunique()}")
    
    return temporal_pbr_data


# ============================================================================
# PANEL A: DISEASE CIS WITH BOOTSTRAP CIs
# ============================================================================

def calculate_disease_cis_bootstrap(temporal_pbr_data, n_bootstrap=1000):
    """
    Calculate Gini decomposition CIS on the fly using country-resampled bootstrap.
    Calculated at the Country-Disease-Year Level.
    """
    print("\n[GRAPH] Calculating disease CIS at CDY level with bootstrap CIs...")
    
    unique_countries = temporal_pbr_data['ISO3'].unique()
    
    # Speedup: group by ISO3 beforehand to avoid repetitive filtering in the loop
    country_groups = {c: df for c, df in temporal_pbr_data.groupby('ISO3')}
    
    boot_cis = {d: [] for d in custom_diseases}
    np.random.seed(42)
    
    for boot_i in range(n_bootstrap):
        # Resample countries with replacement
        boot_countries = np.random.choice(unique_countries, size=len(unique_countries), replace=True)
        
        # Build resampled dataframe
        boot_df = pd.concat([country_groups[c] for c in boot_countries], ignore_index=True)
        
        # Calculate overall Gini
        pbr_all = boot_df['PBR'].values
        gini_all = calculate_gini_coefficient(pbr_all)
        
        # Calculate Gini when each disease is excluded
        for disease in custom_diseases:
            reduced_pbr = boot_df[boot_df['Disease'] != disease]['PBR'].values
            gini_red = calculate_gini_coefficient(reduced_pbr)
            
            cis = (gini_all - gini_red) / gini_all * 100 if gini_all > 0 else 0
            boot_cis[disease].append(cis)

    ci_results = []
    for d in custom_diseases:
        vals = boot_cis[d]
        ci_results.append({
            'Disease': d,
            'CIS_Mean': np.mean(vals),
            'CIS_CI_Lower': np.percentile(vals, 2.5),
            'CIS_CI_Upper': np.percentile(vals, 97.5)
        })
        
    ci_df = pd.DataFrame(ci_results).sort_values('CIS_Mean', ascending=True)
    
    print("[OK] Finished disease CIS bootstrap.")
    return ci_df


# ============================================================================
# PANEL B: LORENZ CURVES FOR DISEASE REMOVAL
# ============================================================================

def calculate_disease_lorenz_curves(temporal_pbr_data, top_diseases):
    """Calculate Lorenz curves with and without top CIS diseases at CDY level"""
    print("\n[GRAPH] Calculating disease-level Lorenz curves at CDY level...")
    
    all_pvals = temporal_pbr_data['PBR'].values
    reduced_pvals = temporal_pbr_data[~temporal_pbr_data['Disease'].isin(top_diseases)]['PBR'].values

    x_all, y_all = lorenz_curve_data(all_pvals)
    x_reduced, y_reduced = lorenz_curve_data(reduced_pvals)

    gini_all = calculate_gini_coefficient(all_pvals)
    gini_reduced = calculate_gini_coefficient(reduced_pvals)
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
# PANEL C: TEMPORAL INEQUALITY REDUCTION (SI PLOT DATA)
# ============================================================================

def calculate_temporal_inequality_reduction(temporal_pbr_data, top_diseases):
    """Calculate temporal Gini trends over time with and without top diseases at CDY level"""
    print("\n[GRAPH] Calculating temporal Gini trends at CDY level...")
    
    temporal_pbr_data['Period'] = (temporal_pbr_data['YEAR'] // 2) * 2
    periods = sorted(temporal_pbr_data['Period'].unique())

    results = []
    for period in periods:
        period_data = temporal_pbr_data[temporal_pbr_data['Period'] == period]

        if len(period_data) < 30:
            continue

        pbr_all = period_data['PBR'].values
        gini_all = calculate_gini_coefficient(pbr_all)

        period_reduced = period_data[~period_data['Disease'].isin(top_diseases)]
        if len(period_reduced) > 0:
            pbr_reduced = period_reduced['PBR'].values
            gini_reduced = calculate_gini_coefficient(pbr_reduced)
        else:
            gini_reduced = gini_all

        results.append({
            'Period': period,
            'Gini_All': gini_all,
            'Gini_Reduced': gini_reduced,
            'Inequality_Reduction': ((gini_all - gini_reduced) / gini_all * 100) if gini_all > 0 else 0
        })

    return pd.DataFrame(results)


# ============================================================================
# PANEL C (MAIN) / D (SI): BETWEEN VS WITHIN DISEASE TEMPORAL TRENDS
# ============================================================================

def calculate_theil_decomposition(period_agg):
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
    between_pct = (between_component / total_theil * 100) if total_theil > 0 else 0
    within_pct = (within_component / total_theil * 100) if total_theil > 0 else 0

    return {
        'total_theil': total_theil,
        'between_percent': between_pct,
        'within_percent': within_pct,
        'between_component': between_component,
        'within_component': within_component
    }


def calculate_temporal_theil_trends(temporal_pbr_data, n_bootstrap=100):
    """Calculate temporal trends in Theil decomposition with bootstrap confidence intervals"""
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
        period_agg = period_agg[~np.isinf(period_agg['PBR']) & (period_agg['PBR'] > 0)]

        if len(period_agg) < 30:
            continue

        theil_result = calculate_theil_decomposition(period_agg)

        results.append({
            'Period': period,
            'Between_Percent': theil_result['between_percent'],
            'Within_Percent': theil_result['within_percent'],
            'Total_Theil': theil_result['total_theil']
        })

        # Bootstrap for confidence intervals
        bootstrap_between = []
        bootstrap_within = []
        unique_countries = period_agg['ISO3'].unique()

        for boot_i in range(n_bootstrap):
            boot_countries = np.random.choice(unique_countries, size=len(unique_countries), replace=True)
            boot_data = []
            for country in boot_countries:
                boot_data.append(period_agg[period_agg['ISO3'] == country])

            if len(boot_data) > 0:
                boot_period_agg = pd.concat(boot_data, ignore_index=True)
                boot_theil = calculate_theil_decomposition(boot_period_agg)
                bootstrap_between.append(boot_theil['between_percent'])
                bootstrap_within.append(boot_theil['within_percent'])

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

    if len(bootstrap_df) > 0:
        results_df = results_df.merge(bootstrap_df, on='Period', how='left')

    return results_df


# ============================================================================
# PANEL D: COUNTRY-LEVEL LORENZ CURVES FOR COUNTRY REMOVAL
# ============================================================================

def calculate_country_lorenz_curves(temporal_pbr_data):
    """Calculate Lorenz curves with and without top 20% countries by participation volume"""
    print("\n[GRAPH] Calculating country-level Lorenz curves...")
    
    # Aggregate total participants to country level
    country_totals = temporal_pbr_data.groupby('ISO3')['Total_Participants'].sum().sort_values(ascending=False)
    all_participants = country_totals.values

    # Top 20% countries to remove
    n_countries_to_remove = max(1, int(len(country_totals) * 0.20))
    top_countries = country_totals.head(n_countries_to_remove).index.tolist()

    print(f"   Removing top {n_countries_to_remove} countries: {', '.join(top_countries[:5])}...")

    # Without top countries
    reduced_participants = country_totals.drop(top_countries).values

    x_all, y_all = lorenz_curve_data(all_participants)
    x_reduced, y_reduced = lorenz_curve_data(reduced_participants)

    gini_all = calculate_gini_coefficient(all_participants)
    gini_reduced = calculate_gini_coefficient(reduced_participants)
    reduction_pct = ((gini_all - gini_reduced) / gini_all * 100) if gini_all > 0 else 0

    print(f"   Gini all countries: {gini_all:.4f}")
    print(f"   Gini without top 20%: {gini_reduced:.4f}")
    print(f"   Reduction: {reduction_pct:.2f}%")

    return {
        'x_all': x_all, 'y_all': y_all, 'gini_all': gini_all,
        'x_reduced': x_reduced, 'y_reduced': y_reduced, 'gini_reduced': gini_reduced,
        'reduction_pct': reduction_pct,
        'n_removed': n_countries_to_remove,
        'top_countries': top_countries
    }


# ============================================================================
# VISUALIZATION LAYOUT
# ============================================================================

def generate_and_save_figure2(include_panel_c=False):
    """Generate and save Figure 2 (4 panels or 5 panels)"""
    print("\n" + "="*80)
    if include_panel_c:
        print("GENERATING SUPPLEMENTARY FIGURE 2 (5 PANELS)")
    else:
        print("GENERATING MAIN FIGURE 2 (4 PANELS)")
    print("="*80)

    temporal_pbr_data = load_and_prepare_temporal_pbr_data()
    ci_results = calculate_disease_cis_bootstrap(temporal_pbr_data, n_bootstrap=1000)

    # Sort and identify top 3 driver diseases (top 20% of 16 diseases is ~3 diseases)
    ci_results_sorted = ci_results.sort_values('CIS_Mean', ascending=False)
    top_3_diseases = ci_results_sorted.head(3)['Disease'].tolist()

    print("\nTop 3 driver diseases by CIS:")
    for i, d in enumerate(top_3_diseases, 1):
        mean_val = ci_results[ci_results['Disease'] == d]['CIS_Mean'].values[0]
        print(f"  {i}. {d}: {mean_val:.3f}%")

    disease_lorenz = calculate_disease_lorenz_curves(temporal_pbr_data, top_3_diseases)
    temporal_reduction = calculate_temporal_inequality_reduction(temporal_pbr_data, top_3_diseases)
    temporal_theil = calculate_temporal_theil_trends(temporal_pbr_data, n_bootstrap=1000)
    country_lorenz = calculate_country_lorenz_curves(temporal_pbr_data)

    # Set up gridspec
    if include_panel_c:
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ========================================================================
    # PANEL A: Disease CIS Bar Chart
    # ========================================================================
    if include_panel_c:
        ax1 = fig.add_subplot(gs[0, 0:2])
    else:
        ax1 = fig.add_subplot(gs[0, 0])

    y_pos = range(len(ci_results))
    ax1.barh(y_pos, ci_results['CIS_Mean'], color='#CC6699', alpha=0.6)
    ax1.errorbar(ci_results['CIS_Mean'], y_pos,
                 xerr=[ci_results['CIS_Mean'] - ci_results['CIS_CI_Lower'],
                       ci_results['CIS_CI_Upper'] - ci_results['CIS_Mean']],
                 fmt='none', color='black', capsize=3, linewidth=1.5, alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Contribution to Inequality Score (CIS) with 95% CI', fontsize=12, fontweight='bold')
    ax1.set_title('A. Disease Contribution to Global Inequality', fontsize=14, fontweight='bold', pad=15)
    ax1.set_yticks(y_pos)
    labels = []
    for d in ci_results['Disease']:
        if d == 'HIV/AIDS and sexually transmitted infections':
            labels.append('HIV/AIDS and\nsexually transmitted infections')
        elif d == 'Neglected tropical diseases and malaria':
            labels.append('Neglected tropical\ndiseases and malaria')
        elif d == 'Respiratory infections and tuberculosis':
            labels.append('Respiratory infections\nand tuberculosis')
        else:
            labels.append(d)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # ========================================================================
    # PANEL B: Disease Lorenz Curves
    # ========================================================================
    if include_panel_c:
        ax2 = fig.add_subplot(gs[0, 2:4])
    else:
        ax2 = fig.add_subplot(gs[0, 1])

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

    ax2.set_xlabel('Cumulative Share of Country-Disease-Year Pairs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Share of PBR', fontsize=12, fontweight='bold')
    ax2.set_title('B. Disease-Level Inequality Reduction', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ========================================================================
    # PANEL C (SI only): Temporal reduction line plot
    # ========================================================================
    if include_panel_c:
        ax3 = fig.add_subplot(gs[0, 4:6])
        from scipy.interpolate import UnivariateSpline

        periods = temporal_reduction['Period'].values
        gini_all = temporal_reduction['Gini_All'].values
        gini_reduced = temporal_reduction['Gini_Reduced'].values

        if len(periods) >= 4:
            period_smooth = np.linspace(periods.min(), periods.max(), 100)
            try:
                spline_all = UnivariateSpline(periods, gini_all, s=len(periods) * 0.02, k=3)
                spline_reduced = UnivariateSpline(periods, gini_reduced, s=len(periods) * 0.02, k=3)
                ax3.plot(period_smooth, spline_all(period_smooth), '-', color=COLORS['primary_blue'], linewidth=3, alpha=0.8, label='All Diseases')
                ax3.plot(period_smooth, spline_reduced(period_smooth), '-', color=COLORS['primary_red'], linewidth=3, alpha=0.8, label='Top 20% Removed')
                ax3.scatter(periods, gini_all, color=COLORS['primary_blue'], s=60, alpha=0.6, zorder=3, edgecolors='white')
                ax3.scatter(periods, gini_reduced, color=COLORS['primary_red'], s=60, alpha=0.6, zorder=3, edgecolors='white', marker='s')
            except:
                ax3.plot(periods, gini_all, 'o-', color=COLORS['primary_blue'], linewidth=2.5, label='All Diseases')
                ax3.plot(periods, gini_reduced, 's-', color=COLORS['primary_red'], linewidth=2.5, label='Top 20% Removed')
        else:
            ax3.plot(periods, gini_all, 'o-', color=COLORS['primary_blue'], linewidth=2.5, label='All Diseases')
            ax3.plot(periods, gini_reduced, 's-', color=COLORS['primary_red'], linewidth=2.5, label='Top 20% Removed')

        # Fit simple trendlines
        if len(temporal_reduction) > 2:
            slope_a, int_a, _, _, _ = stats.linregress(periods, gini_all)
            ax3.plot(periods, slope_a * periods + int_a, '--', color=COLORS['primary_blue'], alpha=0.5)
            slope_r, int_r, _, _, _ = stats.linregress(periods, gini_reduced)
            ax3.plot(periods, slope_r * periods + int_r, '--', color=COLORS['primary_red'], alpha=0.5)

        ax3.set_xlabel('Period (2-year bins)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
        ax3.set_title('C. Temporal Inequality Evolution\n(Removing Top 20% Diseases)', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(2000, 2026, 4))
        ax3.set_ylim(0, 1.0)

    # ========================================================================
    # PANEL C (MAIN) / D (SI): Between vs Within Disease Temporal Trends
    # ========================================================================
    if include_panel_c:
        ax4 = fig.add_subplot(gs[1, 1:3])
        panel_title = 'D. Between-Disease vs Within-Disease Inequality'
    else:
        ax4 = fig.add_subplot(gs[1, 0])
        panel_title = 'C. Temporal Decomposition of Inequality'

    periods = temporal_theil['Period'].values
    between_pct = temporal_theil['Between_Percent'].values
    within_pct = temporal_theil['Within_Percent'].values

    from scipy.interpolate import UnivariateSpline
    if len(periods) >= 4:
        period_smooth = np.linspace(periods.min(), periods.max(), 100)
        try:
            # Between CI
            if 'Between_CI_Lower' in temporal_theil.columns:
                spl_b_low = UnivariateSpline(periods, temporal_theil['Between_CI_Lower'].values, s=len(periods)*1.5, k=3)
                spl_b_up = UnivariateSpline(periods, temporal_theil['Between_CI_Upper'].values, s=len(periods)*1.5, k=3)
                ax4.fill_between(period_smooth, spl_b_low(period_smooth), spl_b_up(period_smooth), color='#CC6699', alpha=0.3, label='Between 95% CI')
            
            # Within CI
            if 'Within_CI_Lower' in temporal_theil.columns:
                spl_w_low = UnivariateSpline(periods, temporal_theil['Within_CI_Lower'].values, s=len(periods)*1.5, k=3)
                spl_w_up = UnivariateSpline(periods, temporal_theil['Within_CI_Upper'].values, s=len(periods)*1.5, k=3)
                ax4.fill_between(period_smooth, spl_w_low(period_smooth), spl_w_up(period_smooth), color='#CC9966', alpha=0.3, label='Within 95% CI')

            # Mean lines
            spl_b = UnivariateSpline(periods, between_pct, s=len(periods)*2, k=3)
            spl_w = UnivariateSpline(periods, within_pct, s=len(periods)*2, k=3)
            ax4.plot(period_smooth, spl_b(period_smooth), '-', color='#CC6699', linewidth=3, label='Between-Disease (smooth)')
            ax4.plot(period_smooth, spl_w(period_smooth), '-', color='#CC9966', linewidth=3, label='Within-Disease (smooth)')
        except:
            ax4.plot(periods, between_pct, 'o-', color='#CC6699', linewidth=2.5, label='Between-Disease')
            ax4.plot(periods, within_pct, 's-', color='#CC9966', linewidth=2.5, label='Within-Disease')
    else:
        ax4.plot(periods, between_pct, 'o-', color='#CC6699', linewidth=2.5, label='Between-Disease')
        ax4.plot(periods, within_pct, 's-', color='#CC9966', linewidth=2.5, label='Within-Disease')

    ax4.scatter(periods, between_pct, color='#CC6699', s=60, zorder=3)
    ax4.scatter(periods, within_pct, color='#CC9966', s=60, zorder=3)

    # Linear trendlines
    slope_b, int_b, r_b, p_b, _ = stats.linregress(periods, between_pct)
    ax4.plot(periods, slope_b * periods + int_b, '--', color='#CC6699', alpha=0.5)
    slope_w, int_w, r_w, p_w, _ = stats.linregress(periods, within_pct)
    ax4.plot(periods, slope_w * periods + int_w, '--', color='#CC9966', alpha=0.5)

    ax4.set_xlabel('Period Start Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% of Total Inequality', fontsize=12, fontweight='bold')
    ax4.set_title(panel_title, fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(range(2000, 2026, 4))
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=9, loc='center right')
    ax4.grid(True, alpha=0.3)

    ax4.text(0.02, 0.98,
             f"Between: {slope_b:.2f}%/period (R²={r_b**2:.3f}, p={p_b:.3f})\nWithin: {slope_w:.2f}%/period (R²={r_w**2:.3f}, p={p_w:.3f})",
             transform=ax4.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # ========================================================================
    # PANEL D (MAIN) / E (SI): Country Lorenz Curves
    # ========================================================================
    if include_panel_c:
        ax5 = fig.add_subplot(gs[1, 3:5])
        panel_title_5 = 'E. Country-Level Inequality Reduction'
    else:
        ax5 = fig.add_subplot(gs[1, 1])
        panel_title_5 = 'D. Country-Level Inequality Reduction'

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
    ax5.set_title(panel_title_5, fontsize=14, fontweight='bold', pad=15)
    ax5.legend(fontsize=11, loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # Save
    plt.tight_layout()
    output_dir = "/Users/wen/Desktop/participation_inequality/analysiswoold"
    
    if include_panel_c:
        png_out = os.path.join(output_dir, "figure2_supplementary_5panels.png")
        pdf_out = os.path.join(output_dir, "figure2_supplementary_5panels.pdf")
    else:
        png_out = os.path.join(output_dir, "figure2.png")
        pdf_out = os.path.join(output_dir, "figure2.pdf")

    plt.savefig(png_out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_out, bbox_inches='tight', facecolor='white')
    print(f"[OK] Figure successfully saved to:\n  - {png_out}\n  - {pdf_out}")

    # Save supplementary data files
    if include_panel_c:
        temporal_reduction.to_csv(os.path.join(output_dir, 'figure2_panel_c_temporal_reduction.csv'), index=False)
        temporal_theil.to_csv(os.path.join(output_dir, 'figure2_panel_d_theil_decomposition.csv'), index=False)
        print("[OK] Supplementary data files saved.")

    plt.close()


def main():
    # Generate 4-panel Main Figure 2
    generate_and_save_figure2(include_panel_c=False)
    
    # Generate 5-panel Supplementary version to write the CSV files
    generate_and_save_figure2(include_panel_c=True)


if __name__ == "__main__":
    main()
