import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

def format_number(val):
    if val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    else:
        return f"{int(val)}"

def generate_figure():
    # Define file paths
    csv_path = "/Users/wen/Desktop/participation_inequality/analysiswoold/geoinfor195kwoold.csv"
    year_path = "/Users/wen/Desktop/participation_inequality/data/year_195k.csv"
    all_about_path = "/Users/wen/Desktop/participation_inequality/data/AllAboutCountry.csv"
    output_dir = "/Users/wen/Desktop/participation_inequality/analysiswoold"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print("Reading data files...")
    df_geo = pd.read_csv(csv_path).rename(columns={'PMID': 'pmid', 'Amount': 'amount', 'ISO3': 'iso3'})
    df_year = pd.read_csv(year_path).rename(columns={'PMID': 'pmid'})
    df_all = pd.read_csv(all_about_path)
    
    # Merge geographic data with publication year
    df = df_geo.merge(df_year, on='pmid', how='inner')
    df['YEAR'] = df['YEAR'].astype(int)
    
    # Filter to unified 180 countries matching strict intersection
    #unified_countries = pd.read_csv("/Users/wen/Desktop/participation_inequality/analysiswoold/unified_180_countries.csv")['ISO3'].unique()
    #df = df[df['iso3'].isin(unified_countries)]
    
    # Clean income classifications
    income_df = df_all[df_all['Type'] == 'Income'].copy()
    income_df['Year'] = pd.to_numeric(income_df['Year'], errors='coerce')
    income_df = income_df.dropna(subset=['Year'])
    income_df['Year'] = income_df['Year'].astype(int)
    
    income_map = {
        'H': 'High income',
        'UM': 'Upper middle income',
        'LM': 'Lower middle income',
        'L': 'Low income'
    }
    income_df['Income_Group'] = income_df['Value'].map(income_map)
    
    # Correct swaps in AllAboutCountry (South Korea KOR should be High income, North Korea PRK should be Low income)
    income_df.loc[income_df['ISO3'] == 'KOR', 'Income_Group'] = 'High income'
    income_df.loc[income_df['ISO3'] == 'PRK', 'Income_Group'] = 'Low income'
    
    # Create complete grid (2000-2024) to map yearly income groups (handling 2024)
    iso3s = income_df['ISO3'].unique()
    years = list(range(2000, 2025))
    grid = pd.MultiIndex.from_product([iso3s, years], names=['ISO3', 'Year']).to_frame().reset_index(drop=True)
    income_grid = grid.merge(income_df[['ISO3', 'Year', 'Income_Group']], on=['ISO3', 'Year'], how='left')
    income_grid = income_grid.sort_values(['ISO3', 'Year'])
    income_grid['Income_Group'] = income_grid.groupby('ISO3')['Income_Group'].ffill().bfill()
    
    # Dict for missing countries/territories
    missing_countries_map = {
        'JEY': 'High income', 'SMR': 'High income', 'MTQ': 'High income', 'ATA': 'High income',
        'ABW': 'High income', 'DMA': 'Upper middle income', 'IMN': 'High income', 'ANT': 'High income',
        'AND': 'High income', 'MDV': 'Upper middle income', 'GUM': 'High income', 'GLP': 'High income',
        'PLW': 'Upper middle income', 'PSE': 'Lower middle income', 'REU': 'High income', 'MCO': 'High income',
        'MNP': 'High income', 'COM': 'Low income', 'ATG': 'High income', 'LIE': 'High income', 'MRT': 'Lower middle income'
    }
    
    # Map income to trial data
    df = df.merge(income_grid[['ISO3', 'Year', 'Income_Group']], left_on=['iso3', 'YEAR'], right_on=['ISO3', 'Year'], how='left')
    
    # Fill missing values from hardcoded dict or latest available year
    hardcoded_s = df['iso3'].map(missing_countries_map)
    df['Income_Group'] = df['Income_Group'].fillna(hardcoded_s)
    
    latest_income = income_df.sort_values('Year').drop_duplicates('ISO3', keep='last')
    fallback_s = df['iso3'].map(dict(zip(latest_income['ISO3'], latest_income['Income_Group'])))
    df['Income_Group'] = df['Income_Group'].fillna(fallback_s)
    
    # Clean population data
    pop_df = df_all[df_all['Type'] == 'Population'].copy()
    pop_df['Year'] = pd.to_numeric(pop_df['Year'], errors='coerce')
    pop_df = pop_df.dropna(subset=['Year'])
    pop_df['Year'] = pop_df['Year'].astype(int)
    pop_df['Value'] = pd.to_numeric(pop_df['Value'], errors='coerce')
    pop_grid = pop_df.pivot(index='ISO3', columns='Year', values='Value').ffill(axis=1).bfill(axis=1)
    
    # Populations by mean across years
    pop_mean = pop_grid.mean(axis=1).to_dict()
    
    # ----------------------------------------------------
    # STATISTICAL TESTING
    # ----------------------------------------------------
    print("\n--- Performing Statistical Analysis ---")
    
    # Calculate country-level total participants and participation rate (per million)
    country_stats = df.groupby(['iso3', 'Income_Group'])['amount'].sum().reset_index()
    country_stats['pop'] = country_stats['iso3'].map(pop_mean)
    country_stats = country_stats.dropna(subset=['pop'])
    country_stats['rate'] = (country_stats['amount'] / country_stats['pop']) * 1e6
    
    # Kruskal-Wallis Test
    kw_groups = [g['rate'].values for name, g in country_stats.groupby('Income_Group')]
    kw_stat, kw_p = stats.kruskal(*kw_groups)
    print(f"Kruskal-Wallis Test: H = {kw_stat:.3f}, p = {kw_p:.4e}")
    
    # Spearman Correlation
    income_rank = {'Low income': 1, 'Lower middle income': 2, 'Upper middle income': 3, 'High income': 4}
    country_stats['rank'] = country_stats['Income_Group'].map(income_rank)
    sp_rho, sp_p = stats.spearmanr(country_stats['rank'], country_stats['rate'])
    print(f"Spearman Correlation: rho = {sp_rho:.3f}, p = {sp_p:.4e}")
    
    # Averages by Income Level
    avg_rates = country_stats.groupby('Income_Group')['rate'].mean()
    print("\nAverage participation rates (participants per million):")
    for group_name in ['High income', 'Upper middle income', 'Lower middle income', 'Low income']:
        print(f"  {group_name}: {avg_rates.get(group_name, 0):,.1f}")
        
    # Chi-square Test: crosstabulation of multi-country vs income group at study level
    study_n_countries = df.groupby('pmid')['iso3'].nunique()
    study_level = df.drop_duplicates('pmid').copy()
    study_level['multi_country'] = study_level['pmid'].map(study_n_countries) > 1
    
    contingency = pd.crosstab(study_level['Income_Group'], study_level['multi_country'])
    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test (Income Group vs Study Type): chi2 = {chi2_stat:.3f}, p = {chi2_p:.4e}")
    
    # ----------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------
    print("\nGenerating Figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    
    # Define color palette: Dark Green to Dark Red
    colors = {
        'High income': '#1e4620',       # Dark green
        'Upper middle income': '#74c69d', # Light green
        'Lower middle income': '#f4a261', # Orange
        'Low income': '#d90429'          # Dark red
    }
    
    order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    
    # Panel a: Temporal trends in total participants recruited (2000-2024)
    panel_a_data = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2024)]
    yearly_sums = panel_a_data.groupby(['Income_Group', 'YEAR'])['amount'].sum().reset_index()
    
    for group_name in order:
        group_data = yearly_sums[yearly_sums['Income_Group'] == group_name]
        ax1.plot(
            group_data['YEAR'],
            group_data['amount'],
            marker='o',
            linewidth=2.5,
            markersize=6,
            color=colors[group_name],
            label=group_name
        )
        
    ax1.set_title("(a) Temporal trends in total participants recruited by income level (2000–2024)", loc='left', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Total Participants Recruited (Millions)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    ax1.set_xlim(1999, 2025)
    ax1.set_xticks(range(2000, 2025, 4))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title="Income Classification", frameon=True, loc='upper left', fontsize=10)
    
    # Panel b: Cumulative participants in multi-country studies by income level
    multi_studies_pmids = study_n_countries[study_n_countries > 1].index
    df_multi = df[df['pmid'].isin(multi_studies_pmids)]
    
    multi_sums = df_multi.groupby('Income_Group')['amount'].sum().reindex(order).fillna(0)
    
    bars = ax2.bar(
        x=np.arange(len(order)),
        height=multi_sums.values,
        color=[colors[name] for name in order],
        edgecolor='black',
        width=0.6,
        alpha=0.85
    )
    
    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        label = f"{format_number(height)}"
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + (multi_sums.max() * 0.02),
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        
    ax2.set_title("(b) Cumulative participants in multi-country studies by income level", loc='left', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(np.arange(len(order)))
    ax2.set_xticklabels(order, fontsize=10)
    ax2.set_ylabel("Cumulative Participants", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    ax2.set_ylim(0, multi_sums.max() * 1.12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figures
    png_out = os.path.join(output_dir, "extended_data_fig2.png")
    pdf_out = os.path.join(output_dir, "extended_data_fig2.pdf")
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    print(f"Figures successfully saved to:\n  - {png_out}\n  - {pdf_out}")

if __name__ == '__main__':
    generate_figure()
