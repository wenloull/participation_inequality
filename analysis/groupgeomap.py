# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Color palettes from your specification
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
#
# # Your 16 custom diseases for analysis
# custom_diseases = [
#     'HIV/AIDS and sexually transmitted infections',
#     'Neglected tropical diseases and malaria',
#     'Maternal and neonatal disorders',
#     'Nutritional deficiencies',
#     'Respiratory infections and tuberculosis',
#     'Chronic respiratory diseases',
#     'Digestive diseases',
#     'Mental disorders',
#     'Neurological disorders',
#     'Cardiovascular diseases',
#     'Diabetes and kidney diseases',
#     'Musculoskeletal disorders',
#     'Neoplasms',
#     'Sense organ diseases',
#     'Skin and subcutaneous diseases',
#     'Substance use disorders'
# ]
#
# print("🔄 RECALCULATING PBR WITH FULL 2000-2024 DATA RANGE")
# print("=" * 60)
#
# # Load datasets
# print("Loading datasets...")
# pmid_cause_70k = pd.read_csv('pmid_cause_70k.csv')
# geoinfor = pd.read_csv('geoinfor.csv')
# disease_mapping = pd.read_csv('disease_mapping.csv')
# country_mapping = pd.read_csv('country_mapping_for_figure.csv')
# gbd_disease = pd.read_csv('gbddisease.csv')
# year_70k = pd.read_csv('year_70k.csv')
#
# print(f"✅ Datasets loaded successfully!")
#
# # Fix disease inheritance
# print("Fixing disease inheritance...")
# parent_mapping = {}
# for _, row in disease_mapping.iterrows():
#     if pd.notna(row['Parent Name']) and row['Parent Name'] in custom_diseases:
#         parent_mapping[row['REI Name']] = row['Parent Name']
#
# inherited_rows = []
# for _, study in pmid_cause_70k.iterrows():
#     disease_name = study['CAUSE']
#     if disease_name in parent_mapping:
#         new_row = study.copy()
#         new_row['CAUSE'] = parent_mapping[disease_name]
#         inherited_rows.append(new_row)
#
# if inherited_rows:
#     inherited_df = pd.DataFrame(inherited_rows)
#     pmid_cause_fixed = pd.concat([pmid_cause_70k, inherited_df], ignore_index=True)
#     print(f"✅ Added {len(inherited_rows)} inherited Level 2 tags")
# else:
#     pmid_cause_fixed = pmid_cause_70k.copy()
#
# # Use FULL 2000-2024 data range (correcting the previous mistake)
# full_years = year_70k[(year_70k['YEAR'] >= 2000) & (year_70k['YEAR'] <= 2024)].copy()
# print(f"📅 Using FULL years 2000-2024: {len(full_years):,} studies")
#
# # Calculate participants by country and disease
# study_data = pmid_cause_fixed.merge(full_years, on='PMID', how='inner')
# print(f"🔗 Studies with year data: {len(study_data):,}")
#
# participant_data = study_data.merge(geoinfor, on='PMID', how='inner')
# print(f"🌍 Studies with participant geo data: {len(participant_data):,}")
#
# participant_data = participant_data[participant_data['CAUSE'].isin(custom_diseases)].copy()
# print(f"🎯 Studies with custom diseases: {len(participant_data):,}")
#
# # First calculate annual totals, then average
# annual_participants = participant_data.groupby(['ISO3', 'CAUSE', 'YEAR']).agg({
#     'Amount': 'sum',
#     'PMID': 'nunique'
# }).reset_index()
#
# country_disease_participants = annual_participants.groupby(['ISO3', 'CAUSE']).agg({
#     'Amount': 'mean',  # Average annual participants
#     'PMID': 'sum'      # Total studies across all years
# }).reset_index()
#
# country_disease_participants.columns = ['ISO3', 'Disease', 'Total_Participants', 'Total_Studies']
# print(f"✅ Participant data: {len(country_disease_participants)} country-disease pairs")
#
# # Calculate DALYs with full data range
# full_dalys = gbd_disease[
#     (gbd_disease['year'] >= 2000) &
#     (gbd_disease['year'] <= 2024) &
#     (gbd_disease['cause_name'].isin(custom_diseases))
#     ].copy()
#
# country_disease_dalys = full_dalys.groupby(['location_name', 'cause_name']).agg({
#     'val': 'mean'
# }).reset_index()
# country_disease_dalys.columns = ['Country', 'Disease', 'Avg_DALYs']
# print(f"✅ DALY data: {len(country_disease_dalys)} country-disease pairs")
#
# # Calculate corrected PBR
# participant_data_mapped = country_disease_participants.merge(
#     country_mapping[['ISO3', 'Standardized']],
#     on='ISO3',
#     how='left'
# )
#
# pbr_data_full = participant_data_mapped.merge(
#     country_disease_dalys,
#     left_on=['Standardized', 'Disease'],
#     right_on=['Country', 'Disease'],
#     how='outer'
# )
#
# pbr_data_full['Total_Participants'] = pbr_data_full['Total_Participants'].fillna(0)
# pbr_data_full['Avg_DALYs'] = pbr_data_full['Avg_DALYs'].fillna(0.1)
#
# print(f"✅ Merged data: {len(pbr_data_full)} country-disease pairs")
#
# # Calculate corrected PBR with full data
# for disease in custom_diseases:
#     disease_mask = pbr_data_full['Disease'] == disease
#     disease_data = pbr_data_full[disease_mask].copy()
#
#     # Only calculate for countries with both participants and DALYs
#     valid_data = disease_data[
#         (disease_data['Total_Participants'] > 0) &
#         (disease_data['Avg_DALYs'] > 0)
#         ].copy()
#
#     if len(valid_data) > 0:
#         total_participants = valid_data['Total_Participants'].sum()
#         total_dalys = valid_data['Avg_DALYs'].sum()
#
#         print(f"🔍 {disease}:")
#         print(f"   Global participants: {total_participants:,.0f}")
#         print(f"   Global DALYs: {total_dalys:,.0f}")
#
#         participant_shares = valid_data['Total_Participants'] / total_participants
#         daly_shares = valid_data['Avg_DALYs'] / total_dalys
#
#         # Apply minimum DALY share threshold and reasonable PBR cap
#         min_daly_share = 0.001
#         adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
#         corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)
#
#         # Update the dataframe
#         valid_indices = valid_data.index
#         pbr_data_full.loc[valid_indices, 'Corrected_PBR'] = corrected_pbr
#         pbr_data_full.loc[valid_indices, 'Participant_Share'] = participant_shares
#         pbr_data_full.loc[valid_indices, 'DALY_Share'] = daly_shares
#         pbr_data_full.loc[valid_indices, 'Corrected_log_PBR'] = np.log10(corrected_pbr)
#
# # Fill missing values
# pbr_data_full['Corrected_PBR'] = pbr_data_full['Corrected_PBR'].fillna(0)
# pbr_data_full['Corrected_log_PBR'] = pbr_data_full['Corrected_log_PBR'].fillna(np.nan)
#
# # Compare with previous calculation
# print(f"\n📈 COMPARISON: 2015-2024 vs 2000-2024 DATA:")
# print("-" * 50)
#
# # Load previous data for comparison
# try:
#     pbr_old = pd.read_csv('pbr_data_corrected.csv')
#     old_valid = pbr_old[pbr_old['Corrected_PBR'] > 0]['Corrected_PBR']
#     new_valid = pbr_data_full[pbr_data_full['Corrected_PBR'] > 0]['Corrected_PBR']
#
#     print(f"Previous (2015-2024): {len(old_valid):,} pairs, Mean PBR: {old_valid.mean():.2f}")
#     print(f"New (2000-2024):      {len(new_valid):,} pairs, Mean PBR: {new_valid.mean():.2f}")
#     print(
#         f"Increase in data:      +{len(new_valid) - len(old_valid):,} pairs ({(len(new_valid) / len(old_valid) - 1) * 100:+.1f}%)")
# except:
#     new_valid = pbr_data_full[pbr_data_full['Corrected_PBR'] > 0]['Corrected_PBR']
#     print(f"New calculation (2000-2024): {len(new_valid):,} pairs, Mean PBR: {new_valid.mean():.2f}")
#
# # Save the corrected data with full date range
# pbr_data_full.to_csv('pbr_data_corrected.csv', index=False)
#
# # Create summary statistics
# print(f"\n📊 FINAL SUMMARY (2000-2024 DATA):")
# print("-" * 40)
# print(f"Total country-disease pairs: {len(pbr_data_full):,}")
# print(f"Pairs with participants: {len(pbr_data_full[pbr_data_full['Corrected_PBR'] > 0]):,}")
# print(f"Mean PBR (non-zero): {new_valid.mean():.2f}")
# print(f"Median PBR (non-zero): {new_valid.median():.2f}")
# print(f"Over-represented (PBR > 1): {len(new_valid[new_valid > 1]):,}")
# print(f"Highly over-represented (PBR > 3): {len(new_valid[new_valid > 3]):,}")
#
# print(f"\n✅ Full 2000-2024 PBR data saved as 'pbr_data_2000_2024_corrected.csv'")
# print(f"🎯 Ready to create corrected world maps with RdBu_r colormap!")

#===================Run above to create the pbr data, Then run this==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

# Load data
pbr_data = pd.read_csv('pbr_data_corrected.csv')

# Your 16 custom diseases
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


def create_world_maps(pbr_data, save_path='rq1_figure1_world_maps.png'):
    """Create clean world maps showing PBR by disease"""

    # Load world shapefile
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    print("World shapefile loaded successfully!")

    # Calculate actual data range but center colormap at 0
    valid_log_pbr = pbr_data[pbr_data['Corrected_log_PBR'].notna()]['Corrected_log_PBR']
    data_min = valid_log_pbr.min()
    data_max = valid_log_pbr.max()

    # Center the colormap at 0 (log PBR = 0 means PBR = 1)
    max_abs = max(abs(data_min), abs(data_max))
    vmin_centered = -max_abs
    vmax_centered = max_abs

    print(f"Data range: {data_min:.2f} to {data_max:.2f}")
    print(f"Using centered range: {vmin_centered:.2f} to {vmax_centered:.2f}")
    print(f"This ensures white = PBR of 1 (proportional)")

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Global Trial Participation-to-Burden Ratio (PBR) by Disease Group\n',
                 fontsize=18, fontweight='bold', y=0.95)

    axes_flat = axes.flatten()
    from matplotlib.colors import LinearSegmentedColormap
    # Create custom colormap: Red -> Gray -> Blue
    colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#e0e0e0',
              '#fee090', '#fdae61', '#f46d43', '#d73027']
    diverging_cmap = LinearSegmentedColormap.from_list('custom_BluGyRed', colors, N=256)

    for i, disease in enumerate(custom_diseases):
        ax = axes_flat[i]

        # Get disease data
        disease_data = pbr_data[
            (pbr_data['Disease'] == disease) &
            (pbr_data['Corrected_PBR'] > 0)
            ].copy()

        # Plot base world map (dark gray for no data)
        world.plot(ax=ax, color='white', edgecolor='gray', linewidth=0.4, alpha=1.0)

        if len(disease_data) > 0:
            # Merge with world shapefile
            world_disease = world.merge(
                disease_data[['ISO3', 'Corrected_log_PBR', 'Corrected_PBR']],
                left_on='iso_a3',
                right_on='ISO3',
                how='left'
            )

            # Plot PBR data
            world_disease.plot(
                ax=ax,
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

            # Annotate over-represented countries (PBR > 3)
            over_rep = disease_data[disease_data['Corrected_PBR'] > 3].nlargest(4, 'Corrected_PBR')
            for _, country in over_rep.iterrows():
                iso3 = country['ISO3']
                country_geom = world[world['iso_a3'] == iso3]
                if not country_geom.empty:
                    try:
                        centroid = country_geom.geometry.centroid.iloc[0]
                        ax.annotate(
                            f"{iso3}\n{country['Corrected_log_PBR']:.2f}",
                            (centroid.x, centroid.y),
                            fontsize=5, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='darkred',
                                      alpha=0.3, edgecolor='red'),
                            color='white'
                        )
                    except:
                        pass

            # Annotate under-represented countries (PBR < 0.5)
            under_rep = disease_data[
                (disease_data['Corrected_PBR'] > 0) &
                (disease_data['Corrected_PBR'] < 0.5)
                ].nsmallest(4, 'Corrected_PBR')
            for _, country in under_rep.iterrows():
                iso3 = country['ISO3']
                country_geom = world[world['iso_a3'] == iso3]
                if not country_geom.empty:
                    try:
                        centroid = country_geom.geometry.centroid.iloc[0]
                        ax.annotate(
                            f"{iso3}\n{country['Corrected_log_PBR']:.2f}",
                            (centroid.x, centroid.y),
                            fontsize=5, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='darkblue',
                                      alpha=0.3, edgecolor='blue'),
                            color='white'
                        )
                    except:
                        pass

        # Styling
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title
        participant_count = disease_data['Total_Participants'].sum() if len(disease_data) > 0 else 0
        country_count = len(disease_data)
        title = disease.replace(' and ', ' &\n') if len(disease) > 35 else disease
        ax.set_title(f'{title}\n({participant_count:,.0f} participants, {country_count} countries)',
                     fontsize=9, fontweight='bold', pad=10)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.015])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap, norm=Normalize(vmin=-1.5, vmax=1.5))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Log₁₀(PBR)', fontsize=12, fontweight='bold', labelpad=10)

    # Add legend
    legend_text = (
        "Participation-to-Burden Ratio (PBR) Interpretation:\n"
        "• Blue: Under-represented (PBR < 1)\n"
        "• Gray: Proportional (PBR ≈ 1)\n"
        "• Red: Over-represented (PBR > 1)\n"
        "• White: No participants\n"
        # And modify the White line to:
        "• Light Gray: Proportional (PBR ≈ 1)\n"
        "• Dark red boxes: Most over-represented (PBR > 3)\n"
        "• Dark blue boxes: Most under-represented (PBR < 0.5)\n\n"
        f"Data: 2000-2024, {pbr_data['ISO3'].nunique()} countries, "
        f"{pbr_data[pbr_data['Corrected_PBR'] > 0]['Total_Participants'].sum():,.0f} total participants"
    )

    fig.text(0.02, 0.02, legend_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.7", facecolor='white', alpha=0.9, edgecolor='gray'))
    fig.patch.set_facecolor('#fefcf6')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.95, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#fefcf6')
    plt.show()

    return fig


# Run the visualization
print("Creating RQ1 World Maps...")
fig = create_world_maps(pbr_data)
print("RQ1 Figure 1 completed!")