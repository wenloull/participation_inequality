"""
Combined Figure: Geographic Inequality in Disease Research - REVISED
- Panel A (top): Geographic maps for CVD and HIV PBR with small legends
- Panel B (bottom left): SI heatmap (3) + boxplot per country colored by income (1) + stacked bar (1)
- Panel C (bottom right): Level 1 scatter plots with symmetrical axes, diagonal lines, and trend lines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
import os
import warnings

warnings.filterwarnings('ignore')

# Color palettes
# COLORS = {
#     'primary_blue': '#2E86AB',
#     'primary_red': '#A23B72',
#     'accent_orange': '#F18F01',
#     'danger_red': '#C73E1D',
#     'info_blue': '#5E7CE2',
#     'warning_orange': '#F4A261',
#     'dark_green': '#264653',
#     'light_yellow': '#E9C46A',
#     'neutral_gray': '#6C757D',
#     'light_gray': '#F8F9FA'
# }

GBD_LEVEL1_COLORS = {
    'Communicable, maternal, neonatal, and nutritional diseases': '#000000',
    'Non-communicable diseases': '#000000',
    'Injuries': '#4682B4'
}

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

custom_disease = [
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

custom_disease_order = [
    'HIV/AIDS and sexually transmitted infections',
    'Neglected tropical diseases and malaria',
    'Respiratory infections and tuberculosis',
    'Maternal and neonatal disorders',
    'Nutritional deficiencies',
    'Chronic respiratory diseases',
    'Digestive diseases',
    'Neoplasms',
    'Cardiovascular diseases',
    'Neurological disorders',
    'Mental disorders',
    'Substance use disorders',
    'Diabetes and kidney diseases',
    'Skin and subcutaneous diseases',
    'Sense organ diseases',
    'Musculoskeletal disorders'
]

def load_data():
    """Load all datasets"""
    print("Loading datasets...")

    base_dir = r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data"
    aggregated_file = os.path.join(base_dir, "public_aggregated_participants_138k.csv")
    gbd_file = os.path.join(base_dir, "gbddisease.csv")
    country_mapping_file = os.path.join(base_dir, "country_mapping_for_figure.csv")

    # Load public aggregated dataset
    print(f"Loading data from {aggregated_file}...")
    try:
        aggregated_data = pd.read_csv(aggregated_file)
    except FileNotFoundError:
        print(f"Error: File {aggregated_file} not found. Please run create_public_dataset.py first.")
        return

    # Load GBD disease burden data
    print(f"Loading data from {gbd_file}...")
    gbddisease = pd.read_csv(gbd_file)

    # Load country mapping
    print(f"Loading data from {country_mapping_file}...")
    country_mapping = pd.read_csv(country_mapping_file)

    # Load AllAboutCountry for income data
    all_about_country_file = os.path.join(base_dir, "AllAboutCountry.csv")
    print(f"Loading data from {all_about_country_file}...")
    all_about_country = pd.read_csv(all_about_country_file, low_memory=False)

    # Load Disease Mapping for hierarchy
    disease_mapping_file = os.path.join(base_dir, "disease_mapping.csv")
    print(f"Loading data from {disease_mapping_file}...")
    disease_mapping = pd.read_csv(disease_mapping_file)

    print("Data loaded successfully!")
    return aggregated_data, gbddisease, country_mapping, all_about_country, disease_mapping

def create_fixed_pmid_cause(disease_mapping, pmid_cause):
    """Create fixed dataset with inherited Level 2 tags"""
    level2_map = {}
    level2_ids = set(disease_mapping[disease_mapping['Level'] == 2]['REI ID'])

    for _, row in disease_mapping[disease_mapping['Level'] == 2].iterrows():
        level2_map[row['REI ID']] = row['REI ID']

    for _, row in disease_mapping[disease_mapping['Level'] == 3].iterrows():
        if row['Parent ID'] in level2_ids:
            level2_map[row['REI ID']] = row['Parent ID']

    for _, row in disease_mapping[disease_mapping['Level'] == 4].iterrows():
        level3_parent = row['Parent ID']
        level3_row = disease_mapping[disease_mapping['REI ID'] == level3_parent]
        if len(level3_row) > 0:
            level2_grandparent = level3_row.iloc[0]['Parent ID']
            if level2_grandparent in level2_ids:
                level2_map[row['REI ID']] = level2_grandparent

    pmid_cause_expanded = pmid_cause.copy()
    pmid_cause_expanded['level2_parent'] = pmid_cause_expanded['cause_id'].map(level2_map)

    level34_mask = pmid_cause_expanded['Level'].isin([3, 4]) & pmid_cause_expanded['level2_parent'].notna()
    level2_inherited = pmid_cause_expanded[level34_mask][['PMID', 'level2_parent']].copy()
    level2_inherited = level2_inherited.rename(columns={'level2_parent': 'cause_id'})
    level2_inherited['Level'] = 2

    pmid_cause_fixed = pd.concat([pmid_cause, level2_inherited], ignore_index=True).drop_duplicates()

    return
    # Prepare data for panel A (already aggregated)
    panel_a_data = aggregated_data.copy()
    
    # Filter for relevant years if needed (already done in aggregation, but good to be safe)
    panel_a_data = panel_a_data[(panel_a_data['YEAR'] >= 2000) & (panel_a_data['YEAR'] <= 2024)]

    create_panel_a_maps(panel_a_data, gbddisease, axes_a)

def create_panel_a_maps(aggregated_data, gbddisease, axes):
    """Create Panel A: CVD and HIV maps using aggregated data"""
    print("\n=== Creating Panel A: Geographic Maps ===")

    # Load world shapefile
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except:
        print("Warning: Could not load world shapefile.")
        return

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.cm as cm
    colors = ['#FF6600', '#FF8C42', '#B0B0B0', '#8FBC45', '#66AA00']
    diverging_cmap = LinearSegmentedColormap.from_list('custom_GreenGreyOrange', colors, N=256)

    diseases = ['Cardiovascular diseases', 'HIV/AIDS and sexually transmitted infections']
    
    for idx, disease_name in enumerate(diseases):
        print(f"Processing {disease_name}...")

        # Filter aggregated data for this disease
        disease_data = aggregated_data[aggregated_data['Disease'] == disease_name].copy()
        disease_data = disease_data[(disease_data['YEAR'] >= 2000) & (disease_data['YEAR'] <= 2024)]

        # Calculate average annual participants per country
        # Aggregated data is per ISO3-Year.
        country_participants = disease_data.groupby('ISO3').agg({
            'Total_Participants': 'mean'
        }).reset_index()

        # Get DALYs for this disease
        # Assuming GBD has 'cause_name' matching 'Disease' or mapped.
        # Check GBD columns. Usually 'cause_name'.
        disease_dalys = gbddisease[gbddisease['cause_name'] == disease_name].copy()
        disease_dalys = disease_dalys[(disease_dalys['year'] >= 2000) & (disease_dalys['year'] <= 2024)]

        # Average DALYs across years per country
        country_dalys = disease_dalys.groupby('ISO3').agg({
            'val': 'mean'
        }).reset_index()
        country_dalys.columns = ['ISO3', 'Avg_DALYs']

        # Merge and calculate PBR
        pbr_data = country_participants.merge(country_dalys, on='ISO3', how='outer')
        pbr_data['Total_Participants'] = pbr_data['Total_Participants'].fillna(0)
        pbr_data['Avg_DALYs'] = pbr_data['Avg_DALYs'].fillna(0.1)

        # Calculate PBR with share-based method
        valid_data = pbr_data[(pbr_data['Total_Participants'] > 0) & (pbr_data['Avg_DALYs'] > 0)].copy()

        if len(valid_data) > 0:
            total_participants = valid_data['Total_Participants'].sum()
            total_dalys = valid_data['Avg_DALYs'].sum()

            participant_shares = valid_data['Total_Participants'] / total_participants
            daly_shares = valid_data['Avg_DALYs'] / total_dalys

            min_daly_share = 0.001
            adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
            corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)

            valid_data['Corrected_PBR'] = corrected_pbr
            valid_data['Corrected_log_PBR'] = np.log10(corrected_pbr)

            pbr_data = pbr_data.merge(valid_data[['ISO3', 'Corrected_log_PBR']], on='ISO3', how='left')
        else:
            pbr_data['Corrected_log_PBR'] = np.nan

        # Merge with world map
        world_disease = world.merge(pbr_data[['ISO3', 'Corrected_log_PBR']],
                                    left_on='iso_a3', right_on='ISO3', how='left')

        # Plot base world map
        world.plot(ax=axes[idx], color='white', edgecolor='gray', linewidth=0.4, alpha=1.0)

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
            linewidth=0.2,
            alpha=0.9
        )


        # Add SMALL colorbar
        # sm = cm.ScalarMappable(cmap=diverging_cmap, norm=Normalize(vmin=-1.5, vmax=1.5))
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=axes[idx], orientation='horizontal',
        #                    shrink=0.3, pad=0.02, aspect=15)
        # cbar.set_label('Log(PBR)', fontsize=7)
        # cbar.ax.tick_params(labelsize=6)
        # Add SINGLE shared colorbar between Panel A and Panel B
        # Calculate position: center between Panel A and Panel B
        fig = axes[0].get_figure()

        # Create a single colorbar for both maps
        sm = cm.ScalarMappable(cmap=diverging_cmap, norm=Normalize(vmin=-1.5, vmax=1.5))
        sm.set_array([])

        # Position colorbar between Panel A and Panel B
        cbar_ax = fig.add_axes([0.5, 0.65, 0.08, 0.01])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Log(PBR)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # Styling
        axes[idx].set_xlim(-180, 180)
        axes[idx].set_ylim(-60, 85)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        title = 'A. Cardiovascular diseases' if idx == 0 else 'B. HIV/AIDS and sexually transmitted infections'
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=5)
        # Remove spines
        for spine in axes[idx].spines.values():
            spine.set_visible(False)

def create_panel_b_heatmap_data(aggregated_data, country_mapping):
    """Create Panel B heatmap data using aggregated data"""
    print("\n=== Creating Panel B: Heatmap Data ===")

    # Filter years
    df = aggregated_data[(aggregated_data['YEAR'] >= 2000) & (aggregated_data['YEAR'] <= 2024)].copy()

    # Filter diseases - use only the 16 diseases in custom_disease_order
    df_filtered = df[df['Disease'].isin(custom_disease_order)].copy()
    
    # Get all regions and find top 10 countries per region
    regions = country_mapping['Region'].unique()
    panel_d_countries = []

    excluded_countries = [
        'BWA', 'GRL', 'ASM', 'FJI', 'FSM', 'MHL', 'PNG', 'SLB', 'VUT', 'WSM',
        'CUB', 'CRI', 'GTM', 'NIC'
    ]

    # Calculate total participants per country (ALL diseases in dataset)
    country_totals = df.groupby('ISO3')['Total_Participants'].sum().reset_index()
    
    for region in regions:
        # Get countries in this region
        region_countries_list = country_mapping[country_mapping['Region'] == region]['ISO3'].tolist()
        
        # Filter country_totals for this region
        region_data = country_totals[country_totals['ISO3'].isin(region_countries_list)]
        
        # Top 10
        top_10 = region_data.nlargest(10, 'Total_Participants')['ISO3'].tolist()
        
        # Filter excluded countries
        filtered_top_10 = [c for c in top_10 if c not in excluded_countries][:10]
        panel_d_countries.extend(filtered_top_10)

    # Calculate SI for selected countries and filtered diseases
    # Tw_d: Total world participants (filtered diseases)
    Tw_d = df_filtered['Total_Participants'].sum()
    
    # Fw: World participants per disease
    Fw = df_filtered.groupby('Disease')['Total_Participants'].sum()
    
    # Calculate SI
    results = []
    
    df_filtered_grouped = df_filtered.groupby(['ISO3', 'Disease'])['Total_Participants'].sum().reset_index()
    country_totals_filtered = df_filtered.groupby('ISO3')['Total_Participants'].sum()
    
    unique_diseases_final = df_filtered['Disease'].unique()
    
    for country in panel_d_countries:
        c_total = country_totals_filtered.get(country, 0)
        
        for disease in unique_diseases_final:
            fw_val = Fw.get(disease, 0)
            
            # Fe: participants for this disease in this country
            fe_row = df_filtered_grouped[(df_filtered_grouped['ISO3'] == country) & (df_filtered_grouped['Disease'] == disease)]
            fe_val = fe_row['Total_Participants'].iloc[0] if not fe_row.empty else 0
            
            if c_total > 0 and fw_val > 0 and Tw_d > 0:
                si = (fe_val / c_total) / (fw_val / Tw_d)
                log_si = np.log10(si) if si > 0 else -3
            else:
                si = 0
                log_si = 0
                
            results.append({
                'Country': country,
                'Disease': disease,
                'SI': si,
                'Log_SI': log_si
            })

    panel_d_df = pd.DataFrame(results)
    
    # Create pivot for heatmap (Disease x Country as per original visual, or Country x Disease?)
    # Original visual seems to be Disease on Y, Country on X?
    # No, original code had `index='Country', columns='Disease'`.
    # But checking Step 252 replacement, I tried `index='Disease', columns='Country'`.
    # Let's check `create_panel_b` logic.
    # `image = axes[1].imshow(panel_d_pivot.values ...)`
    # `axes[1].set_xticks(range(len(panel_d_pivot.columns)))` (X axis = columns)
    # `axes[1].set_yticks(range(len(panel_d_pivot.index)))` (Y axis = index)
    # If we want Disease on X axis (Top Marginal counts "For each disease (X-axis)"), then Columns must be Diseases.
    # So `index='Country', columns='Disease'`.
    
    panel_d_pivot = panel_d_df.pivot(index='Country', columns='Disease', values='Log_SI').fillna(0)
    
    # Ordering
    # Sort diseases by global prevalence (Fw)
    disease_order_d = Fw.sort_values(ascending=False).index.tolist()
    # Ensure they are in columns
    # panel_d_pivot = panel_d_pivot.reindex(columns=disease_order_d) # if columns=Disease
    
    # Sort countries by region (panel_d_countries has them in order)
    country_order_d = panel_d_countries
    
    panel_d_pivot = panel_d_pivot.reindex(index=country_order_d, columns=disease_order_d)
    
    # Get country metadata
    # Replace ISO3 with country names - EXACT from 5_3.py
    country_names = country_mapping.set_index('ISO3')['Standardized'].to_dict()
    panel_d_pivot.index = [country_names.get(iso, iso) for iso in panel_d_pivot.index]
    country_region_map = country_mapping.set_index('ISO3')['Region'].to_dict()
    country_subregion_map = country_mapping.set_index('ISO3')['Subregion'].to_dict()

    return (panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
            country_names, country_region_map, country_subregion_map)

def create_panel_b(panel_d_pivot, panel_d_df, country_order_d, disease_order_d, country_names,
                   country_subregion_map, allaboutcountry, disease_mapping, axes):
    """Create Panel B: marginal bar on top + heatmap + marginal bar on right"""
    print("\n=== Creating Panel B: Three-part visualization ===")

    # Get income data
    income_data = allaboutcountry[allaboutcountry['Type'] == 'Income'].copy()
    income_data = income_data.sort_values('Year', ascending=False).drop_duplicates('ISO3')
    income_data = income_data[['ISO3', 'Value']].rename(columns={'Value': 'Income_Level'})
    income_data['Income_Level'] = income_data['Income_Level'].map(INCOME_CODE_MAP)

    # Create Level 1 mapping for disease colors
    level1_map = {}
    for _, disease in disease_mapping[disease_mapping['Level'] == 2].iterrows():
        parent_id = disease['Parent ID']
        parent_row = disease_mapping[disease_mapping['REI ID'] == parent_id]
        if len(parent_row) > 0:
            level1_map[disease['REI Name']] = parent_row.iloc[0]['REI Name']

    # Part 1: TOP MARGINAL - For each disease (X-axis), count countries by income where Log SI > 0
    # Show as 100% stacked bars
    disease_income_proportions = []

    for disease in panel_d_pivot.columns:
        income_counts = {'High income': 0, 'Upper middle income': 0,
                        'Lower middle income': 0, 'Low income': 0}
        total_countries_with_data = 0

        # For each country (row) in the heatmap
        for country_name in panel_d_pivot.index:
            # Get Log SI value for this country-disease
            log_si_value = panel_d_pivot.loc[country_name, disease]

            if log_si_value > 0:  # Only count if Log SI > 0
                total_countries_with_data += 1

                # Find ISO3 for this country
                country_iso = None
                for iso, name in country_names.items():
                    if name == country_name:
                        country_iso = iso
                        break

                if country_iso:
                    # Get income level
                    income_row = income_data[income_data['ISO3'] == country_iso]
                    if len(income_row) > 0:
                        income_level = income_row.iloc[0]['Income_Level']
                        if income_level in income_counts:
                            income_counts[income_level] += 1

        # Convert to proportions (out of 100%)
        if total_countries_with_data > 0:
            income_proportions = {k: (v / total_countries_with_data) * 100
                                 for k, v in income_counts.items()}
        else:
            income_proportions = {k: 0 for k in income_counts.keys()}

        disease_income_proportions.append(income_proportions)

    # Create 100% stacked bar chart
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

    # CRITICAL: Align with heatmap X-axis
    axes[0].set_xlim(-0.5, len(panel_d_pivot.columns)-0.5)
    axes[0].set_xticks([])
    axes[0].set_ylabel('% Countries', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].set_title('C. Income Distribution (Log(SI)>0)', fontsize=14, fontweight='bold', pad=5)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=12, ncol=1)
    axes[0].tick_params(axis='y', labelsize=12)

    # Part 2: HEATMAP - EXACT CODE FROM 5_3.py
    d_min, d_max = panel_d_pivot.values.min(), panel_d_pivot.values.max()
    max_abs = max(abs(d_min), abs(d_max))
    vmin, vmax = -max_abs, max_abs

    colors = [ '#FF6600', '#FFFFFF','#66AA00']
    custom_cmap = LinearSegmentedColormap.from_list('custom_GreenGreyOrange', colors, N=256)
    im = axes[1].imshow(panel_d_pivot.values, cmap=custom_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('D. Specialization Index (Log Scale)', fontsize=14, fontweight='bold', pad=5, y=-0.4)
    axes[1].set_xticks(range(len(panel_d_pivot.columns)))
    axes[1].set_yticks(range(len(panel_d_pivot.index)))

    # Color x-axis labels by Level 1
    for i, disease in enumerate(panel_d_pivot.columns):
        level1_parent = level1_map.get(disease, 'Unknown')
        color = GBD_LEVEL1_COLORS.get(level1_parent, '#000000')
        axes[1].text(i, len(panel_d_pivot.index), disease, rotation=45, ha='right', va='top',
                    fontsize=12, color=color, transform=axes[1].transData, wrap=True)

    # Color y-axis labels by subregion (alternating)
    current_subregion = None
    use_black = True
    for i, country_name in enumerate(panel_d_pivot.index):
        country_iso = None
        for iso, name in country_names.items():
            if name == country_name:
                country_iso = iso
                break

        subregion = country_subregion_map.get(country_iso, 'Unknown')
        if subregion != current_subregion:
            current_subregion = subregion
            use_black = not use_black

        color = '#000000' if use_black else '#808080'
        axes[1].text(-0.5, i, country_name, ha='right', va='center',
                    fontsize=12, color=color, transform=axes[1].transData, wrap=True)

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Part 3: RIGHT MARGINAL - For each country (Y-axis), count diseases where Log SI > 0 vs < 0
    # Use RdBu_r colors: blue for negative, red for positive
    country_si_counts = []

    for country_name in panel_d_pivot.index:
        # Get all Log SI values for this country (row)
        si_values = panel_d_pivot.loc[country_name].values

        count_positive = np.sum(si_values > 0)  # Red (over-specialized)
        count_negative = np.sum(si_values < 0)  # Blue (under-specialized)

        country_si_counts.append({
            'Positive': count_positive,
            'Negative': count_negative
        })

    counts_df = pd.DataFrame(country_si_counts)
    y_pos = np.arange(len(panel_d_pivot.index))

    # Stacked horizontal bars: Negative (blue) on left, Positive (red) on right
    axes[2].barh(y_pos, -counts_df['Negative'], color='#FF6600', label='Log(SI) < 0', alpha=0.6)
    axes[2].barh(y_pos, counts_df['Positive'], color='#66AA00', label='Log(SI) > 0', alpha=0.6)

    axes[2].set_yticks([])  # No y-axis labels
    axes[2].set_ylim(-0.5, len(panel_d_pivot.index)-0.5)
    axes[2].invert_yaxis()
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_xlabel('# Diseases', fontsize=12)
    axes[2].set_title('E. Log(SI) Direction', fontsize=14, fontweight='bold', pad=5)
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=1)
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].tick_params(axis='x', labelsize=12)


def create_panel_c(aggregated_data, gbddisease, allaboutcountry, disease_mapping, axes):
    """Create Panel C: Four income-level scatter plots with disease-country dots and statistics"""
    print("\n=== Creating Panel C: Income-level Scatter Plots ===")

    # Create Level 1 mapping
    level1_map = {}
    for _, disease in disease_mapping[disease_mapping['Level'] == 2].iterrows():
        parent_id = disease['Parent ID']
        parent_row = disease_mapping[disease_mapping['REI ID'] == parent_id]
        if len(parent_row) > 0:
            level1_map[disease['REI Name']] = parent_row.iloc[0]['REI Name']

    # Get income data
    income_data = allaboutcountry[allaboutcountry['Type'] == 'Income'].copy()
    income_data = income_data.sort_values('Year', ascending=False).drop_duplicates('ISO3')
    income_data = income_data[['ISO3', 'Value']].rename(columns={'Value': 'Income_Level'})
    income_data['Income_Level'] = income_data['Income_Level'].map(INCOME_CODE_MAP)

    # Prepare data using aggregated_data
    df = aggregated_data[(aggregated_data['YEAR'] >= 2000) & (aggregated_data['YEAR'] <= 2024)].copy()
    
    # Calculate avg participants per country-disease
    # Aggregated data is per ISO3-Year-Disease
    # We want average across years (2000-2024)
    avg_participants = df.groupby(['ISO3', 'Disease'])['Total_Participants'].mean().reset_index()
    avg_participants.rename(columns={'Total_Participants': 'Avg_Participants_Per_Year'}, inplace=True)
    
    # Calculate avg DALYs per country-disease (cause_name)
    gbd = gbddisease[(gbddisease['year'] >= 2000) & (gbddisease['year'] <= 2024)].copy()
    avg_dalys = gbd.groupby(['ISO3', 'cause_name'])['val'].mean().reset_index()
    avg_dalys.rename(columns={'val': 'Avg_DALYs_Per_Year', 'cause_name': 'Disease'}, inplace=True)
    
    # Merge
    merged = pd.merge(avg_participants, avg_dalys, on=['ISO3', 'Disease'], how='inner')
    
    # Add Level 1 Category
    merged['Level1_Category'] = merged['Disease'].map(level1_map).fillna('Unknown')
    
    # Filter for relevant diseases (Level 2)
    # merged = merged[merged['Disease'].isin(level1_map.keys())]
    
    # Add Income
    disease_country_df = merged.merge(income_data, on='ISO3', how='left')
    disease_country_df = disease_country_df.dropna(subset=['Income_Level'])
    
    # Calculate log values
    disease_country_df['Log_DALYs'] = np.log10(disease_country_df['Avg_DALYs_Per_Year'])
    disease_country_df['Log_Participants'] = np.log10(disease_country_df['Avg_Participants_Per_Year'])

    # Filter out infinite or NaN values
    disease_country_df = disease_country_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_DALYs', 'Log_Participants'])

    print(f"Total disease-country combinations: {len(disease_country_df)}")

    # Get global min/max for consistent axes across all subplots
    global_min = min(disease_country_df['Log_DALYs'].min(), disease_country_df['Log_Participants'].min())
    global_max = max(disease_country_df['Log_DALYs'].max(), disease_country_df['Log_Participants'].max())

    # Add padding
    data_range = global_max - global_min
    padding = data_range * 0.05
    axis_min = global_min - padding
    axis_max = global_max + padding

    # Plot for each income level
    income_levels = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']

    for idx, income_level in enumerate(income_levels):
        income_data_plot = disease_country_df[disease_country_df['Income_Level'] == income_level]

        print(f"{income_level}: {len(income_data_plot)} disease-country combinations")

        # Set SQUARE axis limits first
        axes[idx].set_xlim(axis_min, axis_max)
        axes[idx].set_ylim(axis_min, axis_max)

        # Plot grey dots for each disease-country combination
        axes[idx].scatter(income_data_plot['Log_DALYs'],
                          income_data_plot['Log_Participants'],
                          c='grey', s=30, alpha=0.6,
                          edgecolors='black', linewidth=0.2, zorder=2)

        # Add trend line and calculate statistics
        if len(income_data_plot) >= 2:
            X = income_data_plot['Log_DALYs'].values.reshape(-1, 1)
            y = income_data_plot['Log_Participants'].values

            reg = LinearRegression().fit(X, y)
            x_trend = np.linspace(axis_min, axis_max, 100)
            y_trend = reg.predict(x_trend.reshape(-1, 1))

            # Calculate R-squared and p-value
            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Calculate p-value
            n = len(X)
            if n > 2:
                # Calculate standard error of the coefficient
                x_mean = np.mean(X)
                se = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((X - x_mean) ** 2))
                t_stat = reg.coef_[0] / se
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            else:
                p_value = 1.0

            # Plot trend line in income-level color
            axes[idx].plot(x_trend, y_trend, color=INCOME_COLORS[income_level],
                           linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            # Add statistics text
            stats_text = f'n={len(income_data_plot)}\nβ={reg.coef_[0]:.3f}\np={p_value:.3f}'
            axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add PROPER diagonal reference line (y = x)
        # axes[idx].plot([axis_min, axis_max], [axis_min, axis_max],
        #                color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # Labels and title
        if idx in [2, 3]:  # Bottom row subplots get x-label
            axes[idx].set_xlabel('Log(Avg DALYs/Year)', fontsize=12, fontweight='bold')
        if idx in [0, 2]:  # Left column subplots get y-label
            axes[idx].set_ylabel('Log(Avg Participants/Year)', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{chr(70 + idx)}. {income_level}', fontsize=14, fontweight='bold', pad=5)
        axes[idx].grid(alpha=0.3)

def main():
    """Create the combined figure"""
    print("="*70)
    print("COMBINED FIGURE: GEOGRAPHIC INEQUALITY")
    print("="*70)

    # Load data
    aggregated_data, gbddisease, country_mapping, allaboutcountry, disease_mapping = load_data()

    # Create figure with GridSpec - VERY TIGHT spacing
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[3, 2],
                          hspace=0.02, wspace=0.15)  # Reduced from 0.3 to 0.15

    # Panel A: Top row, spanning both columns (2 maps side by side)
    gs_a = gs[0, :].subgridspec(1, 2, wspace=0.05)  # Reduced from 0.15
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(2)]

    # Panel B: Bottom left, with 3 parts: marginal bar on TOP + heatmap + bar on RIGHT
    # Create 2x2 grid for Panel B
    gs_b = gs[1, 0].subgridspec(2, 2, height_ratios=[1, 5], width_ratios=[4, 1],
                                hspace=0.05, wspace=0.05)
    # Top marginal bar (spans both columns)
    ax_b_top = fig.add_subplot(gs_b[0, 0])
    # Heatmap (bottom left)
    ax_b_heatmap = fig.add_subplot(gs_b[1, 0])
    # Right bar chart (bottom right)
    ax_b_right = fig.add_subplot(gs_b[1, 1])
    axes_b = [ax_b_top, ax_b_heatmap, ax_b_right]

    # Panel C: Bottom right, with 2 vertical parts
    gs_c = gs[1, 1].subgridspec(2, 2, hspace=0.15, wspace=0.15)  # Changed to 2x2
    axes_c = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(2)]  # Changed to 2x2

    # Create panels
    create_panel_a_maps(aggregated_data, gbddisease, axes_a)

    (panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
     country_names, country_region_map, country_subregion_map) = create_panel_b_heatmap_data(
        aggregated_data, country_mapping)

    create_panel_b(panel_d_pivot, panel_d_df, country_order_d, disease_order_d,
                  country_names, country_subregion_map, allaboutcountry, disease_mapping, axes_b)

    create_panel_c(aggregated_data, gbddisease, allaboutcountry, disease_mapping, axes_c)

    # Save
    plt.savefig('C:/Users/dell/PycharmProjects/nlp2/Analysis/SM/combined_figure_final.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('C:/Users/dell/PycharmProjects/nlp2/Analysis/SM/combined_figure_final.pdf',
                dpi=300, bbox_inches='tight')
    print("\n[OK] Figure saved!")

    plt.show()

if __name__ == "__main__":
    main()