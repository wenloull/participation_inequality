"""
Combined Figure: Geographic Inequality in Disease Research (Figure 1)
- Panel A-B (top): Geographic maps for CVD and RST PBR using orange-gray-green colormap.
- Panel C-E (bottom left): Stacked bar for income distribution (C), Heatmap of Specialization Index (D), and Stacked bar for SI direction (E).
- Panel F-I (bottom right): Scatter plots of Log DALYs vs Log Participants per year for each income level, with regression trend lines and statistical annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# Style settings for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# Colors for GBD Level 1 categories
GBD_LEVEL1_COLORS = {
    'Communicable, maternal, neonatal, and nutritional diseases': '#000000',
    'Non-communicable diseases': '#000000',
    'Injuries': '#4682B4'
}

# Colors for Income classifications
INCOME_COLORS = {
    'High income': '#3973ac',
    'Upper middle income': '#6699CC',
    'Lower middle income': '#CC6699',
    'Low income': '#ac3973'
}

INCOME_CODE_MAP = {
    'H': 'High income',
    'UM': 'Upper middle income',
    'LM': 'Lower middle income',
    'L': 'Low income'
}

custom_disease_order = [
    'HIV/AIDS and sexually transmitted infections',
    'Neglected tropical diseases and malaria',
    'Nutritional deficiencies',
    'Respiratory infections and tuberculosis',
    'Maternal and neonatal disorders',
    'Chronic respiratory diseases',
    'Digestive diseases',
    'Neurological disorders',
    'Mental disorders',
    'Neoplasms',
    'Cardiovascular diseases',
    'Substance use disorders',
    'Diabetes and kidney diseases',
    'Skin and subcutaneous diseases',
    'Sense organ diseases',
    'Musculoskeletal disorders'
]

def format_number(val):
    if val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    else:
        return f"{int(val)}"

def create_panel_a_maps(participant_data, gbddisease, df_cmap, axes):
    """Create Panel A-B: CVD and RST maps using new 195k data"""
    print("\n=== Creating Panel A: Geographic Maps ===")

    # Load world shapefile
    try:
        world_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(world_url)
    except Exception as e:
        print(f"Warning: Could not load world shapefile: {e}")
        return

    # Custom colormap (Orange -> Gray -> Green)
    colors = ['#FF6600', '#FF8C42', '#B0B0B0', '#8FBC45', '#66AA00']
    diverging_cmap = LinearSegmentedColormap.from_list('custom_GreenGreyOrange', colors, N=256)

    diseases = ['Cardiovascular diseases', 'Maternal and neonatal disorders']
    
    for idx, disease_name in enumerate(diseases):
        print(f"Processing {disease_name}...")

        # Filter participant data for this disease
        disease_participants_raw = participant_data[participant_data['CAUSE'] == disease_name]
        
        # Calculate annual totals
        annual_parts = disease_participants_raw.groupby(['ISO3', 'YEAR']).agg({'Amount': 'sum'}).reset_index()
        
        # Average annual participants
        country_parts = annual_parts.groupby('ISO3').agg({'Amount': 'mean'}).reset_index()
        country_parts.columns = ['ISO3', 'Total_Participants']

        # Get DALYs for this disease
        disease_dalys = gbddisease[
            (gbddisease['cause_name'] == disease_name) &
            (gbddisease['year'] >= 2000) &
            (gbddisease['year'] <= 2024)
        ].copy()
        
        country_dalys = disease_dalys.groupby('ISO3').agg({'val': 'mean'}).reset_index()
        country_dalys.columns = ['ISO3', 'Avg_DALYs']

        # Merge
        pbr_data = country_parts.merge(country_dalys, on='ISO3', how='outer')
        pbr_data['Total_Participants'] = pbr_data['Total_Participants'].fillna(0)
        pbr_data['Avg_DALYs'] = pbr_data['Avg_DALYs'].fillna(0.1)

        valid_data = pbr_data[(pbr_data['Total_Participants'] > 0) & (pbr_data['Avg_DALYs'] > 0)].copy()

        if len(valid_data) > 0:
            total_participants = valid_data['Total_Participants'].sum()
            total_dalys = valid_data['Avg_DALYs'].sum()

            participant_shares = valid_data['Total_Participants'] / total_participants
            daly_shares = valid_data['Avg_DALYs'] / total_dalys

            min_daly_share = 0.001
            adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
            corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)

            valid_data['Corrected_log_PBR'] = np.log10(corrected_pbr)
            pbr_data = pbr_data.merge(valid_data[['ISO3', 'Corrected_log_PBR']], on='ISO3', how='left')
        else:
            pbr_data['Corrected_log_PBR'] = np.nan

        # Merge with world map
        iso_col = 'iso_a3' if 'iso_a3' in world.columns else 'ISO_A3'
        world_disease = world.merge(pbr_data[['ISO3', 'Corrected_log_PBR']],
                                    left_on=iso_col, right_on='ISO3', how='left')

        # Plot base world map
        world.plot(ax=axes[idx], color='white', edgecolor='#cccccc', linewidth=0.3, alpha=1.0)

        # Plot PBR data
        world_disease.plot(
            ax=axes[idx],
            column='Corrected_log_PBR',
            cmap=diverging_cmap,
            legend=False,
            missing_kwds={'color': 'white'},
            vmin=-1.5,
            vmax=1.5,
            edgecolor='white',
            linewidth=0.15,
            alpha=0.9
        )

        # Styling
        axes[idx].set_xlim(-180, 180)
        axes[idx].set_ylim(-60, 85)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        title = 'A. Cardiovascular diseases' if idx == 0 else 'B. Maternal and neonatal disorders'
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # Remove spines
        for spine in axes[idx].spines.values():
            spine.set_visible(False)

    # Add shared colorbar
    fig = axes[0].get_figure()
    sm = cm.ScalarMappable(cmap=diverging_cmap, norm=Normalize(vmin=-1.5, vmax=1.5))
    cbar_ax = fig.add_axes([0.46, 0.61, 0.08, 0.01])  # Center position between maps
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\mathrm{Log}_{10}(\mathrm{PBR})$', fontsize=10, fontweight='bold', labelpad=5)
    cbar.ax.tick_params(labelsize=8)

def create_panel_b_heatmap_data(participant_data, df_cmap):
    """Create Panel B heatmap data using new 195k data"""
    print("\n=== Creating Panel B: Heatmap Data ===")

    # Calculate cumulative total participants (sum of Amount across all years) for SI calculation
    df_si_raw = participant_data.groupby(['ISO3', 'CAUSE'])['Amount'].sum().reset_index()
    df_si_raw.columns = ['ISO3', 'Disease', 'Total_Participants']

    # Tw_d: Total world participants in filtered diseases
    Tw_d = df_si_raw['Total_Participants'].sum()
    
    # Fw: World participants per disease
    Fw = df_si_raw.groupby('Disease')['Total_Participants'].sum()
    
    # Calculate total participants per country across all custom diseases
    country_totals_filtered = df_si_raw.groupby('ISO3')['Total_Participants'].sum()

    # Define panel_d_countries manually to match the exact regional grouping, ordering, and exclusions
    panel_d_countries = []
    
    # 1. Asia (9 countries in requested order)
    panel_d_countries.extend(['CHN', 'JPN', 'TWN', 'ISR', 'KOR', 'IND', 'BGD', 'PAK', 'NPL'])
    
    # 2. Africa (top 10)
    panel_d_countries.extend(['ZAF', 'MWI', 'KEN', 'ZMB', 'UGA', 'TZA', 'GHA', 'ZWE', 'ETH', 'BWA'])
    
    # 3. Europe (top 10)
    panel_d_countries.extend(['GBR', 'FIN', 'NLD', 'SWE', 'DNK', 'NOR', 'FRA', 'DEU', 'POL', 'ITA'])
    
    # 4. North America (including Mexico and Puerto Rico under USA and Canada, excluding GRL/BMU)
    panel_d_countries.extend(['USA', 'CAN', 'MEX', 'PRI'])
    
    # 5. South America (10 countries, excluding MEX and CRI)
    panel_d_countries.extend(['BRA', 'ARG', 'COL', 'PER',  'CHL', 'VEN','CRI', 'BHS','HTI'])
    
    # 6. Oceania (only Australia and New Zealand, others deleted)
    panel_d_countries.extend(['AUS', 'NZL'])

    # Calculate SI
    results = []
    for country in panel_d_countries:
        c_total = country_totals_filtered.get(country, 0)
        for disease in custom_disease_order:
            fw_val = Fw.get(disease, 0)
            
            # Fe: participants for this disease in this country
            fe_row = df_si_raw[(df_si_raw['ISO3'] == country) & (df_si_raw['Disease'] == disease)]
            fe_val = fe_row['Total_Participants'].iloc[0] if not fe_row.empty else 0
            
            if c_total > 0 and fw_val > 0 and Tw_d > 0:
                si = (fe_val / c_total) / (fw_val / Tw_d)
                log_si = np.log10(si) if si > 0 else -3
            else:
                si = 0
                log_si = -3
                
            results.append({
                'Country': country,
                'Disease': disease,
                'SI': si,
                'Log_SI': log_si
            })

    panel_d_df = pd.DataFrame(results)
    
    panel_d_pivot = panel_d_df.pivot(index='Country', columns='Disease', values='Log_SI').fillna(-3)
    
    # Sort diseases by global prevalence (Fw)
        # Use custom disease order
    disease_order_d = custom_disease_order

    
    # Sort countries by region (panel_d_countries has them in order)
    country_order_d = panel_d_countries
    
    panel_d_pivot = panel_d_pivot.reindex(index=country_order_d, columns=disease_order_d)
    
    # Get country metadata
    country_names = df_cmap.set_index('ISO3')['Standardized'].to_dict()
    panel_d_pivot.index = [country_names.get(iso, iso) for iso in panel_d_pivot.index]
    country_region_map = df_cmap.set_index('ISO3')['Region'].to_dict()
    country_subregion_map = df_cmap.set_index('ISO3')['Subregion'].to_dict()

    return (panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
            country_names, country_region_map, country_subregion_map)

def create_panel_b(panel_d_pivot, panel_d_df, country_order_d, disease_order_d, country_names,
                   country_subregion_map, income_grid, disease_mapping, axes):
    """Create Panel B: marginal bar on top + heatmap + marginal bar on right"""
    print("\n=== Creating Panel B: Three-part visualization ===")

    # Get latest income level for each country
    latest_income = income_grid.sort_values('Year').drop_duplicates('ISO3', keep='last')
    income_map = dict(zip(latest_income['ISO3'], latest_income['Income_Group']))

    # Create Level 1 mapping for disease colors
    level1_map = {}
    for _, disease in disease_mapping[disease_mapping['Level'] == 2].iterrows():
        parent_id = disease['Parent ID']
        parent_row = disease_mapping[disease_mapping['REI ID'] == parent_id]
        if len(parent_row) > 0:
            level1_map[disease['REI Name']] = parent_row.iloc[0]['REI Name']

    # Part 1: TOP MARGINAL - stacked bars for income distribution where Log(SI) > 0
    disease_income_proportions = []

    for disease in panel_d_pivot.columns:
        income_counts = {'High income': 0, 'Upper middle income': 0,
                        'Lower middle income': 0, 'Low income': 0}
        total_countries_with_data = 0

        for country_name in panel_d_pivot.index:
            log_si_value = panel_d_pivot.loc[country_name, disease]

            if log_si_value > 0:
                total_countries_with_data += 1
                country_iso = None
                for iso, name in country_names.items():
                    if name == country_name:
                        country_iso = iso
                        break

                if country_iso:
                    income_level = income_map.get(country_iso, 'Unknown')
                    if income_level in income_counts:
                        income_counts[income_level] += 1

        if total_countries_with_data > 0:
            income_proportions = {k: (v / total_countries_with_data) * 100
                                 for k, v in income_counts.items()}
        else:
            income_proportions = {k: 0 for k in income_counts.keys()}

        disease_income_proportions.append(income_proportions)

    x_pos = np.arange(len(panel_d_pivot.columns))

    high_prop = [d['High income'] for d in disease_income_proportions]
    um_prop = [d['Upper middle income'] for d in disease_income_proportions]
    lm_prop = [d['Lower middle income'] for d in disease_income_proportions]
    low_prop = [d['Low income'] for d in disease_income_proportions]

    axes[0].bar(x_pos, high_prop, color=INCOME_COLORS['High income'],
               label='High income', alpha=0.8, width=0.8)
    axes[0].bar(x_pos, um_prop, bottom=high_prop,
               color=INCOME_COLORS['Upper middle income'], label='Upper middle income', alpha=0.8, width=0.8)
    axes[0].bar(x_pos, lm_prop, bottom=np.array(high_prop)+np.array(um_prop),
               color=INCOME_COLORS['Lower middle income'], label='Lower middle income', alpha=0.8, width=0.8)
    axes[0].bar(x_pos, low_prop,
               bottom=np.array(high_prop)+np.array(um_prop)+np.array(lm_prop),
               color=INCOME_COLORS['Low income'], label='Low income', alpha=0.8, width=0.8)

    axes[0].set_xlim(-0.5, len(panel_d_pivot.columns)-0.5)
    axes[0].set_xticks([])
    axes[0].set_ylabel('% Countries', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].set_title('C. Income Distribution (Log(SI)>0)', fontsize=14, fontweight='bold', pad=5)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=12, ncol=1)
    axes[0].tick_params(axis='y', labelsize=12)

    # Part 2: HEATMAP (SI log scale)
    d_min, d_max = panel_d_pivot.values.min(), panel_d_pivot.values.max()
    max_abs = max(abs(d_min), abs(d_max))
    vmin, vmax = -max_abs, max_abs

    colors_list = ['#FF6600', '#FFFFFF', '#66AA00']
    custom_cmap = LinearSegmentedColormap.from_list('custom_GreenGreyOrange', colors_list, N=256)
    im = axes[1].imshow(panel_d_pivot.values, cmap=custom_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('D. Specialization Index (Log Scale)', fontsize=14, fontweight='bold', pad=5, y=-0.4)
    axes[1].set_xticks(range(len(panel_d_pivot.columns)))
    axes[1].set_yticks(range(len(panel_d_pivot.index)))

    # Color x-axis labels
    for i, disease in enumerate(panel_d_pivot.columns):
        level1_parent = level1_map.get(disease, 'Unknown')
        color = GBD_LEVEL1_COLORS.get(level1_parent, '#000000')
        axes[1].text(i, len(panel_d_pivot.index), disease, rotation=45, ha='right', va='top',
                    fontsize=12, color=color, transform=axes[1].transData, wrap=True)

    # Color y-axis labels
    current_subregion = None
    use_black = True
    for i, country_name in enumerate(panel_d_pivot.index):
        country_iso = None
        for iso, name in country_names.items():
            if name == country_name:
                country_iso = iso
                break

        # Sequence and color override for Asia countries (China, Japan, Taiwan, Israel, South Korea in light grey; others in black)
        if country_iso in ['CHN', 'JPN', 'TWN', 'ISR', 'KOR','PRI']:
            color = '#808080'  # light grey
        elif country_iso in ['IND', 'BGD', 'PAK', 'NPL','CRI']:
            color = '#000000'  # black
        else:
            subregion = country_subregion_map.get(country_iso, 'Unknown')
            if subregion != current_subregion:
                current_subregion = subregion
                use_black = not use_black
            color = '#000000' if use_black else '#808080'

        axes[1].text(-0.5, i, country_name, ha='right', va='center',
                    fontsize=12, color=color, transform=axes[1].transData, wrap=True)

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Part 3: RIGHT MARGINAL - stacked horizontal bars for positive (SI > 0) vs negative (SI < 0) counts
    country_si_counts = []

    for country_name in panel_d_pivot.index:
        si_values = panel_d_pivot.loc[country_name].values
        count_positive = np.sum(si_values > 0)
        count_negative = np.sum(si_values < 0)

        country_si_counts.append({
            'Positive': count_positive,
            'Negative': count_negative
        })

    counts_df = pd.DataFrame(country_si_counts)
    y_pos = np.arange(len(panel_d_pivot.index))

    axes[2].barh(y_pos, -counts_df['Negative'], color='#FF6600', label='Log(SI) < 0', alpha=0.6)
    axes[2].barh(y_pos, counts_df['Positive'], color='#66AA00', label='Log(SI) > 0', alpha=0.6)

    axes[2].set_yticks([])
    axes[2].set_ylim(-0.5, len(panel_d_pivot.index)-0.5)
    axes[2].invert_yaxis()
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_xlabel('# Diseases', fontsize=12)
    axes[2].set_title('E. Log(SI) Direction', fontsize=14, fontweight='bold', pad=5)
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=1)
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].tick_params(axis='x', labelsize=12)

def create_panel_c(participant_data, gbddisease, income_grid, disease_mapping, axes):
    """Create Panel C (F-I): Four income-level scatter plots with trend lines"""
    print("\n=== Creating Panel C: Income-level Scatter Plots ===")

    # Get latest income levels
    latest_income = income_grid.sort_values('Year').drop_duplicates('ISO3', keep='last')
    income_map = dict(zip(latest_income['ISO3'], latest_income['Income_Group']))

    # Calculate average annual participants per country and disease
    annual_parts = participant_data.groupby(['ISO3', 'CAUSE', 'YEAR']).agg({'Amount': 'sum'}).reset_index()
    avg_participants = annual_parts.groupby(['ISO3', 'CAUSE']).agg({'Amount': 'mean'}).reset_index()
    avg_participants.rename(columns={'Amount': 'Avg_Participants_Per_Year', 'CAUSE': 'Disease'}, inplace=True)

    # Calculate average annual DALYs per country and disease
    gbd = gbddisease[(gbddisease['year'] >= 2000) & (gbddisease['year'] <= 2024)].copy()
    avg_dalys = gbd.groupby(['ISO3', 'cause_name'])['val'].mean().reset_index()
    avg_dalys.rename(columns={'val': 'Avg_DALYs_Per_Year', 'cause_name': 'Disease'}, inplace=True)

    # Merge
    merged = pd.merge(avg_participants, avg_dalys, on=['ISO3', 'Disease'], how='inner')
    
    # Add Income
    merged['Income_Level'] = merged['ISO3'].map(income_map)
    disease_country_df = merged.dropna(subset=['Income_Level'])

    # Calculate log values
    disease_country_df['Log_DALYs'] = np.log10(disease_country_df['Avg_DALYs_Per_Year'])
    disease_country_df['Log_Participants'] = np.log10(disease_country_df['Avg_Participants_Per_Year'])

    # Filter out infinite or NaN values
    disease_country_df = disease_country_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_DALYs', 'Log_Participants'])

    print(f"Total disease-country combinations in scatter: {len(disease_country_df)}")

    # Get global min/max for consistent axes across all subplots
    global_min = min(disease_country_df['Log_DALYs'].min(), disease_country_df['Log_Participants'].min())
    global_max = max(disease_country_df['Log_DALYs'].max(), disease_country_df['Log_Participants'].max())

    # Add padding
    data_range = global_max - global_min
    padding = data_range * 0.05
    axis_min = global_min - padding
    axis_max = global_max + padding

    income_levels = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']

    for idx, income_level in enumerate(income_levels):
        income_data_plot = disease_country_df[disease_country_df['Income_Level'] == income_level]
        print(f"  {income_level}: {len(income_data_plot)} combinations")

        # Set square limits
        axes[idx].set_xlim(axis_min, axis_max)
        axes[idx].set_ylim(axis_min, axis_max)

        # Plot gray dots
        axes[idx].scatter(
            income_data_plot['Log_DALYs'],
            income_data_plot['Log_Participants'],
            c='grey', s=30, alpha=0.6,
            edgecolors='black', linewidth=0.2, zorder=2
        )

        # Add diagonal reference line (y = x)
        axes[idx].plot([axis_min, axis_max], [axis_min, axis_max],
                       color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # Run linear regression
        if len(income_data_plot) >= 2:
            X = income_data_plot['Log_DALYs'].values.reshape(-1, 1)
            y = income_data_plot['Log_Participants'].values

            reg = LinearRegression().fit(X, y)
            x_trend = np.linspace(axis_min, axis_max, 100)
            y_trend = reg.predict(x_trend.reshape(-1, 1))

            # Calculate R-squared and standard error
            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Calculate p-value of coefficient
            n = len(X)
            if n > 2:
                x_mean = np.mean(X)
                se = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((X - x_mean) ** 2))
                t_stat = reg.coef_[0] / se
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            else:
                p_value = 1.0

            # Plot trend line
            axes[idx].plot(x_trend, y_trend, color=INCOME_COLORS[income_level],
                           linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            # Add stats annotation
            stats_text = f'n={len(income_data_plot)}\nβ={reg.coef_[0]:.3f}\np={p_value:.3f}'
            axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels and grid
        if idx in [2, 3]:
            axes[idx].set_xlabel('Log(Avg DALYs/Year)', fontsize=12, fontweight='bold')
        if idx in [0, 2]:
            axes[idx].set_ylabel('Log(Avg Participants/Year)', fontsize=12, fontweight='bold')
            
        axes[idx].set_title(f'{chr(70 + idx)}. {income_level}', fontsize=14, fontweight='bold', pad=5)
        axes[idx].grid(alpha=0.3)

def main():
    print("="*70)
    print("COMBINED FIGURE: GEOGRAPHIC INEQUALITY")
    print("="*70)

    # Paths
    pmid_cause_path = "/Users/wen/Desktop/participation_inequality/CauseClassier/pmid_cause.csv"
    geoinfor_path = "/Users/wen/Desktop/participation_inequality/analysiswoold/geoinfor195kwoold.csv"
    year_path = "/Users/wen/Desktop/participation_inequality/data/year_195k.csv"
    gbd_path = "/Users/wen/Desktop/participation_inequality/data/gbddisease.csv"
    c_map_path = "/Users/wen/Desktop/participation_inequality/data/country_mapping_for_figure.csv"
    all_about_path = "/Users/wen/Desktop/participation_inequality/data/AllAboutCountry.csv"
    disease_mapping_path = "/Users/wen/Desktop/participation_inequality/data/disease_mapping.csv"
    output_dir = "/Users/wen/Desktop/participation_inequality/analysiswoold"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    df_geo = pd.read_csv(geoinfor_path).rename(columns={'pmid': 'PMID', 'iso3': 'ISO3', 'amount': 'Amount'})
    df_year = pd.read_csv(year_path).rename(columns={'PMID': 'PMID', 'YEAR': 'YEAR'})
    df_cause = pd.read_csv(pmid_cause_path)
    df_gbd = pd.read_csv(gbd_path)
    df_cmap = pd.read_csv(c_map_path)
    df_all = pd.read_csv(all_about_path)
    disease_mapping = pd.read_csv(disease_mapping_path)

    # Clean Income Classifications
    income_df = df_all[df_all['Type'] == 'Income'].copy()
    income_df['Year'] = pd.to_numeric(income_df['Year'], errors='coerce')
    income_df = income_df.dropna(subset=['Year'])
    income_df['Year'] = income_df['Year'].astype(int)
    income_df['Income_Group'] = income_df['Value'].map(INCOME_CODE_MAP)

    # Correct swaps
    income_df.loc[income_df['ISO3'] == 'KOR', 'Income_Group'] = 'High income'
    income_df.loc[income_df['ISO3'] == 'PRK', 'Income_Group'] = 'Low income'

    # Build complete grid for income mapping
    iso3s = income_df['ISO3'].unique()
    years = list(range(2000, 2025))
    grid = pd.MultiIndex.from_product([iso3s, years], names=['ISO3', 'Year']).to_frame().reset_index(drop=True)
    income_grid = grid.merge(income_df[['ISO3', 'Year', 'Income_Group']], on=['ISO3', 'Year'], how='left')
    income_grid = income_grid.sort_values(['ISO3', 'Year'])
    income_grid['Income_Group'] = income_grid.groupby('ISO3')['Income_Group'].ffill().bfill()

    missing_countries_map = {
        'JEY': 'High income', 'SMR': 'High income', 'MTQ': 'High income', 'ATA': 'High income',
        'ABW': 'High income', 'DMA': 'Upper middle income', 'IMN': 'High income', 'ANT': 'High income',
        'AND': 'High income', 'MDV': 'Upper middle income', 'GUM': 'High income', 'GLP': 'High income',
        'PLW': 'Upper middle income', 'PSE': 'Lower middle income', 'REU': 'High income', 'MCO': 'High income',
        'MNP': 'High income', 'COM': 'Low income', 'ATG': 'High income', 'LIE': 'High income', 'MRT': 'Lower middle income'
    }

    # Filter cause and year
    df_cause_l2 = df_cause[(df_cause['Level'] == 2) & (df_cause['CAUSE'].isin(custom_disease_order))][['PMID', 'CAUSE']].drop_duplicates().copy()
    df_year_filtered = df_year[(df_year['YEAR'] >= 2000) & (df_year['YEAR'] <= 2024)].copy()

    # Merge trial data
    study_data = df_cause_l2.merge(df_year_filtered, on='PMID', how='inner')
    participant_data = study_data.merge(df_geo, on='PMID', how='inner')

    # Apply global country filters (Delete GRL, BMU, and Oceania countries besides AUS/NZL)
    oceania_countries = df_cmap[df_cmap['Region'] == 'Oceania']['ISO3'].unique()
    oceania_to_keep = {'AUS', 'NZL'}
    oceania_to_delete = set(oceania_countries) - oceania_to_keep
    countries_to_delete = {'GRL', 'BMU'} | oceania_to_delete
    
    participant_data = participant_data[~participant_data['ISO3'].isin(countries_to_delete)]
    df_gbd = df_gbd[~df_gbd['ISO3'].isin(countries_to_delete)]

    # Filter to unified 180 countries matching strict intersection
    unified_countries = pd.read_csv("/Users/wen/Desktop/participation_inequality/analysiswoold/unified_180_countries.csv")['ISO3'].unique()
    participant_data = participant_data[participant_data['ISO3'].isin(unified_countries)]
    df_gbd = df_gbd[df_gbd['ISO3'].isin(unified_countries)]

    # Map Income Group to trial data
    participant_data = participant_data.merge(income_grid[['ISO3', 'Year', 'Income_Group']], left_on=['ISO3', 'YEAR'], right_on=['ISO3', 'Year'], how='left')
    hardcoded_s = participant_data['ISO3'].map(missing_countries_map)
    participant_data['Income_Group'] = participant_data['Income_Group'].fillna(hardcoded_s)
    latest_income = income_df.sort_values('Year').drop_duplicates('ISO3', keep='last')
    fallback_s = participant_data['ISO3'].map(dict(zip(latest_income['ISO3'], latest_income['Income_Group'])))
    participant_data['Income_Group'] = participant_data['Income_Group'].fillna(fallback_s)

    # Create figure with GridSpec - TIGHT spacing
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[3.2, 2],
                          hspace=0.08, wspace=0.15)

    # Panel A: Top row, spanning both columns (2 maps side by side)
    gs_a = gs[0, :].subgridspec(1, 2, wspace=0.05)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(2)]

    # Panel B: Bottom left
    gs_b = gs[1, 0].subgridspec(2, 2, height_ratios=[1, 5], width_ratios=[4.5, 1],
                                hspace=0.02, wspace=0.02)
    ax_b_top = fig.add_subplot(gs_b[0, 0])
    ax_b_heatmap = fig.add_subplot(gs_b[1, 0])
    ax_b_right = fig.add_subplot(gs_b[1, 1])
    axes_b = [ax_b_top, ax_b_heatmap, ax_b_right]

    # Panel C: Bottom right
    gs_c = gs[1, 1].subgridspec(2, 2, hspace=0.2, wspace=0.2)
    axes_c = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(2)]

    # Create panels
    create_panel_a_maps(participant_data, df_gbd, df_cmap, axes_a)

    (panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
     country_names, country_region_map, country_subregion_map) = create_panel_b_heatmap_data(
        participant_data, df_cmap)

    create_panel_b(panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
                  country_names, country_subregion_map, income_grid, disease_mapping, axes_b)

    create_panel_c(participant_data, df_gbd, income_grid, disease_mapping, axes_c)

    # Save
    png_out = os.path.join(output_dir, "figure1.png")
    pdf_out = os.path.join(output_dir, "figure1.pdf")
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    
    print(f"\n[OK] Figure successfully saved to:\n  - {png_out}\n  - {pdf_out}")

if __name__ == "__main__":
    main()
