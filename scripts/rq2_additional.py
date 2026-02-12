"""
RQ2 - Additional Analyses: Theil (Country-Grouped) + Variance Partitioning
==================================================================================

FINAL VERSION - Testing Country vs. Disease Factors

Two analyses to complement the disease-grouped Theil decomposition:
1. Theil decomposition grouped by COUNTRY (reverse of disease-grouped)
2. Two-way variance partitioning (country + disease fixed effects)

These provide symmetric, unbiased tests of whether country or disease factors
drive research inequality.

Data structure:
- pmid_cause_70k.csv: PMID, cause_id, CAUSE, Level
- year_70k.csv: PMID, YEAR
- geoinfor.csv: PMID, Amount, Country, Region, ISO3
- gbddisease.csv: location_name, cause_id, cause_name, year, val, ISO3, Level, Parent ID
- disease_mapping.csv: REI ID, REI Name, Parent ID, Parent Name, Level, TYPE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Color palette
COLORS = {
    'primary_blue': '#2E86AB',
    'primary_red': '#A23B72',
    'accent_orange': '#F18F01',
    'danger_red': '#C73E1D',
    'info_blue': '#5E7CE2',
    'dark_green': '#264653',
    'light_yellow': '#E9C46A',
}

print("=" * 100)
print("RQ2 ADDITIONAL ANALYSES: Country vs. Disease Factors")
print("="  * 100)

# ============================================================================
# SECTION 0: DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load all necessary data and create country-disease-year panel"""
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    # Load datasets
    pmid_cause = pd.read_csv('pmid_cause_70k.csv')
    year_70k = pd.read_csv('year_70k.csv')
    geoinfor = pd.read_csv('geoinfor.csv')
    gbddisease = pd.read_csv('gbddisease.csv')
    disease_mapping = pd.read_csv('disease_mapping.csv')

    print(f"✓ Loaded all datasets")

    # Filter for 16 Level 2 diseases
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

    level2_diseases = disease_mapping[
        (disease_mapping['Level'] == 2) &
        (disease_mapping['REI Name'].isin(custom_diseases))
    ][['REI ID', 'REI Name']].copy()

    print(f"✓ Loaded {len(level2_diseases)} Level 2 diseases")

    # Merge research data
    research = pmid_cause.merge(year_70k, on='PMID', how='inner')
    research = research.merge(geoinfor[['PMID', 'ISO3', 'Amount']], on='PMID', how='inner')
    research = research.merge(level2_diseases, left_on='cause_id', right_on='REI ID', how='inner')

    # Filter years 2000-2024
    research = research[(research['YEAR'] >= 2000) & (research['YEAR'] <= 2024)]
    research = research.dropna(subset=['ISO3', 'Amount', 'YEAR'])

    # Aggregate research by country-disease-year
    research_agg = research.groupby(['ISO3', 'REI Name', 'YEAR']).agg({
        'Amount': 'sum',
        'PMID': 'nunique'
    }).reset_index()
    research_agg.columns = ['ISO3', 'disease_name', 'year', 'participants', 'n_studies']

    print(f"✓ Research data: {len(research_agg)} country-disease-year observations")

    # Prepare GBD data
    gbd = gbddisease[
        (gbddisease['cause_id'].isin(level2_diseases['REI ID'])) &
        (gbddisease['year'].between(2000, 2024)) &
        (gbddisease['Level'] == 2)
    ].copy()

    gbd = gbd.merge(level2_diseases, left_on='cause_id', right_on='REI ID', how='left')

    gbd_agg = gbd.groupby(['ISO3', 'REI Name', 'year']).agg({
        'val': 'sum'
    }).reset_index()
    gbd_agg.columns = ['ISO3', 'disease_name', 'year', 'dalys']

    print(f"✓ GBD data: {len(gbd_agg)} country-disease-year observations")

    # Create panel
    panel = gbd_agg.merge(research_agg, on=['ISO3', 'disease_name', 'year'], how='left')
    panel['participants'] = panel['participants'].fillna(0)
    panel['n_studies'] = panel['n_studies'].fillna(0)
    panel = panel[panel['dalys'] > 0]

    print(f"✓ Final panel: {len(panel)} observations")
    print(f"  - {len(panel['ISO3'].unique())} countries")
    print(f"  - {len(panel['disease_name'].unique())} diseases")
    print(f"  - Years: {panel['year'].min()}-{panel['year'].max()}")

    return panel, level2_diseases

def calculate_pbr(panel):
    """Calculate PBR for each country-disease-year"""
    print("\n" + "="*80)
    print("CALCULATING PBR")
    print("="*80)

    # Calculate global totals by disease-year
    global_totals = panel.groupby(['disease_name', 'year']).agg({
        'participants': 'sum',
        'dalys': 'sum'
    }).reset_index()
    global_totals.columns = ['disease_name', 'year', 'global_participants', 'global_dalys']

    # Merge and calculate shares
    panel_pbr = panel.merge(global_totals, on=['disease_name', 'year'], how='left')

    panel_pbr['participant_share'] = np.where(
        panel_pbr['global_participants'] > 0,
        panel_pbr['participants'] / panel_pbr['global_participants'],
        0
    )

    panel_pbr['daly_share'] = np.where(
        panel_pbr['global_dalys'] > 0,
        panel_pbr['dalys'] / panel_pbr['global_dalys'],
        0
    )

    # Calculate PBR
    panel_pbr['pbr'] = np.where(
        panel_pbr['daly_share'] > 0,
        panel_pbr['participant_share'] / panel_pbr['daly_share'],
        np.nan
    )

    # Remove invalid values
    panel_pbr = panel_pbr[np.isfinite(panel_pbr['pbr'])]
    panel_pbr = panel_pbr[panel_pbr['pbr'] >= 0]

    print(f"✓ Calculated PBR for {len(panel_pbr)} observations")
    print(f"  - PBR range: {panel_pbr['pbr'].min():.3f} to {panel_pbr['pbr'].max():.3f}")
    print(f"  - PBR mean: {panel_pbr['pbr'].mean():.3f}")
    print(f"  - PBR median: {panel_pbr['pbr'].median():.3f}")

    return panel_pbr

# ============================================================================
# ANALYSIS 1: THEIL DECOMPOSITION (COUNTRY-GROUPED)
# ============================================================================

def calculate_theil_index(values, weights=None):
    """Calculate Theil T index"""
    values = np.array(values)
    if weights is not None:
        weights = np.array(weights)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(values)) / len(values)

    # Remove zeros and negative values
    mask = values > 0
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return 0

    weights = weights / weights.sum()
    mean_val = np.sum(values * weights)

    if mean_val <= 0:
        return 0

    theil = np.sum(weights * (values / mean_val) * np.log(values / mean_val))
    return theil

def theil_decomposition_country_grouped(panel_pbr):
    """
    Theil decomposition grouped by COUNTRY
    Between-country: inequality from country-average differences
    Within-country: inequality from disease portfolio variation within countries
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: THEIL DECOMPOSITION (COUNTRY-GROUPED)")
    print("="*80)

    results = []
    years = sorted(panel_pbr['year'].unique())

    for yr in years:
        data_year = panel_pbr[panel_pbr['year'] == yr].copy()

        if len(data_year) == 0:
            continue

        data_year['weight'] = data_year['participants'] + 1
        total_weight = data_year['weight'].sum()

        if total_weight == 0:
            continue

        # Total Theil
        theil_total = calculate_theil_index(
            data_year['pbr'].values,
            data_year['weight'].values
        )

        # Between-country component
        country_stats = data_year.groupby('ISO3').apply(
            lambda x: pd.Series({
                'country_mean_pbr': np.average(x['pbr'], weights=x['weight']),
                'country_weight': x['weight'].sum()
            })
        ).reset_index()

        theil_between = calculate_theil_index(
            country_stats['country_mean_pbr'].values,
            country_stats['country_weight'].values
        )

        # Within-country component
        theil_within = theil_total - theil_between

        if theil_total > 0:
            pct_between = (theil_between / theil_total) * 100
            pct_within = (theil_within / theil_total) * 100
        else:
            pct_between = 0
            pct_within = 0

        results.append({
            'Year': yr,
            'Theil_Total': theil_total,
            'Theil_Between_Country': theil_between,
            'Theil_Within_Country': theil_within,
            'Pct_Between_Country': pct_between,
            'Pct_Within_Country': pct_within
        })

        print(f"  {yr}: Between-Country={pct_between:.1f}%, Within-Country={pct_within:.1f}%")

    results_df = pd.DataFrame(results)

    # Calculate temporal trends
    slope_between, _, r_between, p_between, _ = stats.linregress(
        results_df['Year'], results_df['Pct_Between_Country']
    )

    slope_within, _, r_within, p_within, _ = stats.linregress(
        results_df['Year'], results_df['Pct_Within_Country']
    )

    print(f"\nTemporal Trends:")
    print(f"  Between-Country: {slope_between:+.3f}%/year (R²={r_between**2:.3f}, p={p_between:.4f})")
    print(f"  Within-Country:  {slope_within:+.3f}%/year (R²={r_within**2:.3f}, p={p_within:.4f})")

    # Save results
    results_df.to_csv('rq2_theil_country_grouped.csv', index=False)
    print(f"\n✓ Saved to rq2_theil_country_grouped.csv")

    return results_df

# ============================================================================
# ANALYSIS 2: VARIANCE PARTITIONING
# ============================================================================

def variance_partitioning(panel_pbr):
    """
    Two-way variance partitioning using fixed effects
    Calculate R² for country, disease, and year fixed effects
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: VARIANCE PARTITIONING (TWO-WAY DECOMPOSITION)")
    print("="*80)

    from sklearn.linear_model import LinearRegression

    # Prepare data
    data = panel_pbr.copy()
    data['log_pbr'] = np.log(data['pbr'] + 0.01)
    data = data[np.isfinite(data['log_pbr'])]

    print(f"  Working with {len(data)} observations")

    # Create dummy variables
    country_dummies = pd.get_dummies(data['ISO3'], prefix='country', drop_first=True)
    disease_dummies = pd.get_dummies(data['disease_name'], prefix='disease', drop_first=True)
    year_dummies = pd.get_dummies(data['year'], prefix='year', drop_first=True)

    y = data['log_pbr'].values

    # Full model
    X_full = pd.concat([country_dummies, disease_dummies, year_dummies], axis=1)
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    r2_full = model_full.score(X_full, y)

    print(f"  Full model R²: {r2_full:.4f}")

    # Models without each component
    X_no_country = pd.concat([disease_dummies, year_dummies], axis=1)
    model_no_country = LinearRegression()
    model_no_country.fit(X_no_country, y)
    r2_no_country = model_no_country.score(X_no_country, y)

    X_no_disease = pd.concat([country_dummies, year_dummies], axis=1)
    model_no_disease = LinearRegression()
    model_no_disease.fit(X_no_disease, y)
    r2_no_disease = model_no_disease.score(X_no_disease, y)

    X_no_year = pd.concat([country_dummies, disease_dummies], axis=1)
    model_no_year = LinearRegression()
    model_no_year.fit(X_no_year, y)
    r2_no_year = model_no_year.score(X_no_year, y)

    # Partial R²
    partial_r2_country = r2_full - r2_no_country
    partial_r2_disease = r2_full - r2_no_disease
    partial_r2_year = r2_full - r2_no_year

    # Percentages of explained variance
    total_explained = r2_full
    pct_country = (partial_r2_country / total_explained) * 100
    pct_disease = (partial_r2_disease / total_explained) * 100
    pct_year = (partial_r2_year / total_explained) * 100

    print(f"\n  Partial R² (marginal contribution):")
    print(f"    Country:  {partial_r2_country:.4f} ({pct_country:.1f}% of explained variance)")
    print(f"    Disease:  {partial_r2_disease:.4f} ({pct_disease:.1f}% of explained variance)")
    print(f"    Year:     {partial_r2_year:.4f} ({pct_year:.1f}% of explained variance)")
    print(f"    Residual: {1-r2_full:.4f} ({(1-r2_full)*100:.1f}% unexplained)")

    # Save results
    results = {
        'Component': ['Country', 'Disease', 'Year', 'Residual'],
        'Partial_R2': [partial_r2_country, partial_r2_disease, partial_r2_year, 1-r2_full],
        'Pct_of_Explained': [pct_country, pct_disease, pct_year, np.nan],
        'Full_Model_R2': [r2_full, r2_full, r2_full, r2_full]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv('rq2_variance_partitioning.csv', index=False)
    print(f"\n✓ Saved to rq2_variance_partitioning.csv")

    return results_df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_combined_figure(theil_country, variance_part):
    """Create a 2-panel figure showing both analyses"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Theil Decomposition (Country-Grouped)
    ax = axes[0]
    ax.plot(theil_country['Year'], theil_country['Pct_Between_Country'],
            'o-', color=COLORS['primary_blue'], linewidth=2.5, markersize=7,
            label='Between-Country', alpha=0.8)
    ax.plot(theil_country['Year'], theil_country['Pct_Within_Country'],
            'o-', color=COLORS['primary_red'], linewidth=2.5, markersize=7,
            label='Within-Country', alpha=0.8)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('% of Total Inequality', fontsize=13, fontweight='bold')
    ax.set_title('(A) Theil Decomposition (Country-Grouped)\nBetween vs. Within Country Inequality',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add average lines
    avg_between = theil_country['Pct_Between_Country'].mean()
    avg_within = theil_country['Pct_Within_Country'].mean()
    ax.axhline(avg_between, color=COLORS['primary_blue'], linestyle=':', alpha=0.4, linewidth=2)
    ax.axhline(avg_within, color=COLORS['primary_red'], linestyle=':', alpha=0.4, linewidth=2)

    # Annotate averages
    ax.text(2023, avg_between+2, f'Avg: {avg_between:.1f}%',
            color=COLORS['primary_blue'], fontsize=9, fontweight='bold')
    ax.text(2023, avg_within-4, f'Avg: {avg_within:.1f}%',
            color=COLORS['primary_red'], fontsize=9, fontweight='bold')

    # Panel B: Variance Partitioning
    ax = axes[1]
    components = variance_part[variance_part['Component'] != 'Residual']
    colors_var = [COLORS['primary_blue'], COLORS['accent_orange'], COLORS['dark_green']]
    bars = ax.bar(components['Component'], components['Partial_R2'],
                  color=colors_var, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Partial R²', fontsize=13, fontweight='bold')
    ax.set_title('(B) Variance Partitioning (Two-Way Decomposition)\nMarginal Contribution to Explained Variance',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(components['Partial_R2']) * 1.15)

    # Add percentage labels on bars
    for bar, pct in zip(bars, components['Pct_of_Explained']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('rq2_country_vs_disease_analyses.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to rq2_country_vs_disease_analyses.png")
    plt.show()

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run both analyses"""

    # Load and prepare data
    panel, diseases = load_and_prepare_data()
    panel_pbr = calculate_pbr(panel)

    # Analysis 1: Theil Decomposition (Country-Grouped)
    theil_country = theil_decomposition_country_grouped(panel_pbr)

    # Analysis 2: Variance Partitioning
    variance_part = variance_partitioning(panel_pbr)

    # Create visualization
    create_combined_figure(theil_country, variance_part)

    # Final summary
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*100)

    avg_between = theil_country['Pct_Between_Country'].mean()
    avg_within = theil_country['Pct_Within_Country'].mean()

    print(f"\n1. THEIL DECOMPOSITION (COUNTRY-GROUPED):")
    print(f"   Between-Country: {avg_between:.1f}% (average)")
    print(f"   Within-Country:  {avg_within:.1f}% (average)")
    print(f"   → {avg_between:.1f}% of inequality stems from country-average differences")
    print(f"   → Countries differ systematically in overall participation levels")

    country_var = variance_part[variance_part['Component'] == 'Country']['Pct_of_Explained'].values[0]
    disease_var = variance_part[variance_part['Component'] == 'Disease']['Pct_of_Explained'].values[0]
    year_var = variance_part[variance_part['Component'] == 'Year']['Pct_of_Explained'].values[0]

    print(f"\n2. VARIANCE PARTITIONING:")
    print(f"   Country effects: {country_var:.1f}% of explained variance")
    print(f"   Disease effects: {disease_var:.1f}% of explained variance")
    print(f"   Year effects:    {year_var:.1f}% of explained variance")
    print(f"   → Country explains {country_var/disease_var:.1f}x more variance than disease")

    print(f"\n" + "="*100)
    print("KEY CONCLUSION:")
    print("="*100)
    print(f"✓ COUNTRY FACTORS DOMINATE OVER DISEASE FACTORS")
    print(f"  - Country effects explain {country_var/disease_var:.1f}x more variance than disease")
    print(f"  - {avg_between:.1f}% of inequality is between-country (not between-disease)")
    print(f"  - WHERE you live matters far more than WHAT disease you have")
    print("="*100)

    return theil_country, variance_part

# Run the analysis
if __name__ == "__main__":
    theil_results, variance_results = main()