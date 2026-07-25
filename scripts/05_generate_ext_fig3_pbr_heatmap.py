import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

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
    pmid_cause_path = "/Users/wen/Desktop/participation_inequality/CauseClassier/pmid_cause.csv"
    geoinfor_path = "/Users/wen/Desktop/participation_inequality/public/geoinfor183_disease_matched.csv"
    year_path = "/Users/wen/Desktop/participation_inequality/data/year_195k.csv"
    gbd_path = "/Users/wen/Desktop/participation_inequality/data/gbddisease.csv"
    c_map_path = "/Users/wen/Desktop/participation_inequality/data/country_mapping_for_figure.csv"
    output_dir = "/Users/wen/Desktop/participation_inequality/public"
    os.makedirs(output_dir, exist_ok=True)
    
    # 16 Custom diseases (excluding enteric infections)
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
    
    print("Loading datasets...")
    df_geo = pd.read_csv(geoinfor_path).rename(columns={'pmid': 'PMID', 'iso3': 'ISO3', 'amount': 'Amount'})
    df_year = pd.read_csv(year_path).rename(columns={'PMID': 'PMID', 'YEAR': 'YEAR'})
    df_cause = pd.read_csv(pmid_cause_path).rename(columns={'CAUSE': 'Cause'})
    df_gbd = pd.read_csv(gbd_path)
    df_cmap = pd.read_csv(c_map_path)
    
    # Filter pmid_cause to Level 2 and custom_diseases
    df_cause_l2 = df_cause[(df_cause['Level'] == 2) & (df_cause['Cause'].isin(custom_diseases))][['PMID', 'Cause']].drop_duplicates().copy()
    
    # Filter years to 2000-2024
    df_year_filtered = df_year[(df_year['YEAR'] >= 2000) & (df_year['YEAR'] <= 2024)].copy()
    
    # Merge trial data
    study_data = df_cause_l2.merge(df_year_filtered, on='PMID', how='inner')
    participant_data = study_data.merge(df_geo, on='PMID', how='inner')
    
    # Calculate cumulative total participants (raw sum) for title annotation
    cumulative_participants = participant_data.groupby('Cause')['Amount'].sum().to_dict()
    
    # Group by ISO3, CAUSE (which is 'Cause'), and YEAR to calculate annual totals
    annual_participants = participant_data.groupby(['ISO3', 'Cause', 'YEAR']).agg({
        'Amount': 'sum',
        'PMID': 'nunique'
    }).reset_index()
    
    # Average annual participants and total studies
    country_disease_participants = annual_participants.groupby(['ISO3', 'Cause']).agg({
        'Amount': 'mean',  # Average annual participants
        'PMID': 'sum'      # Total studies
    }).reset_index()
    
    country_disease_participants.columns = ['ISO3', 'Disease', 'Total_Participants', 'Total_Studies']
    
    # Calculate GBD average DALYs (2000-2024)
    full_dalys = df_gbd[
        (df_gbd['year'] >= 2000) &
        (df_gbd['year'] <= 2024) &
        (df_gbd['cause_name'].isin(custom_diseases))
    ].copy()
    
    country_disease_dalys = full_dalys.groupby(['location_name', 'cause_name']).agg({
        'val': 'mean'
    }).reset_index()
    country_disease_dalys.columns = ['Country', 'Disease', 'Avg_DALYs']
    
    # Merge participant and DALY data
    participant_data_mapped = country_disease_participants.merge(
        df_cmap[['ISO3', 'Standardized']],
        on='ISO3',
        how='left'
    )
    
    pbr_data = participant_data_mapped.merge(
        country_disease_dalys,
        left_on=['Standardized', 'Disease'],
        right_on=['Country', 'Disease'],
        how='outer'
    )
    
    # Fallback mappings for missing ISO3 codes
    gbd_to_iso = dict(zip(df_gbd['location_name'], df_gbd['ISO3']))
    pbr_data['ISO3'] = pbr_data['ISO3'].fillna(pbr_data['Country'].map(gbd_to_iso))
    pbr_data['Standardized'] = pbr_data['Standardized'].fillna(pbr_data['Country'])
    
    pbr_data['Total_Participants'] = pbr_data['Total_Participants'].fillna(0)
    pbr_data['Total_Studies'] = pbr_data['Total_Studies'].fillna(0)
    pbr_data['Avg_DALYs'] = pbr_data['Avg_DALYs'].fillna(0.1)
    
    # Calculate PBR for each disease
    pbr_data['Corrected_PBR'] = np.nan
    pbr_data['Participant_Share'] = np.nan
    pbr_data['DALY_Share'] = np.nan
    pbr_data['Corrected_log_PBR'] = np.nan
    
    for disease in custom_diseases:
        disease_mask = pbr_data['Disease'] == disease
        disease_data = pbr_data[disease_mask].copy()
        
        # Valid channels
        valid_data = disease_data[
            (disease_data['Total_Participants'] > 0) &
            (disease_data['Avg_DALYs'] > 0)
        ].copy()
        
        if len(valid_data) > 0:
            total_participants = valid_data['Total_Participants'].sum()
            total_dalys = valid_data['Avg_DALYs'].sum()
            
            participant_shares = valid_data['Total_Participants'] / total_participants
            daly_shares = valid_data['Avg_DALYs'] / total_dalys
            
            min_daly_share = 0.001
            adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
            corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)
            
            valid_indices = valid_data.index
            pbr_data.loc[valid_indices, 'Corrected_PBR'] = corrected_pbr
            pbr_data.loc[valid_indices, 'Participant_Share'] = participant_shares
            pbr_data.loc[valid_indices, 'DALY_Share'] = daly_shares
            pbr_data.loc[valid_indices, 'Corrected_log_PBR'] = np.log10(corrected_pbr)
            
    pbr_data['Corrected_PBR'] = pbr_data['Corrected_PBR'].fillna(0)
    pbr_data['Corrected_log_PBR'] = pbr_data['Corrected_log_PBR'].fillna(np.nan)
    
    # Save the PBR data
    pbr_csv_out = os.path.join(output_dir, "pbr_data_2000_2024_corrected.csv")
    pbr_data.to_csv(pbr_csv_out, index=False)
    print(f"Corrected PBR data saved to:\n  - {pbr_csv_out}")
    
    # ----------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------
    print("Loading world map shapefile...")
    world_geojson = "/Users/wen/Desktop/participation_inequality/data/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(world_geojson)
    
    # Setup subplots
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Global Trial Participation-to-Burden Ratio (PBR) by Disease Group (2000–2024)\n',
                 fontsize=20, fontweight='bold', y=0.96)
    
    axes_flat = axes.flatten()
    
    # Define custom diverging colormap (Blue -> Gray -> Red)
    colors_hex = [
        '#08519c', '#3182bd', '#6baed6', '#9ecae1', '#e0e0e0',
        '#fee090', '#fdae61', '#f46d43', '#d73027'
    ]
    diverging_cmap = LinearSegmentedColormap.from_list('custom_BluGyRed', colors_hex, N=256)
    
    iso_col = 'iso_a3' if 'iso_a3' in world.columns else 'ISO_A3'
    
    for i, disease in enumerate(custom_diseases):
        ax = axes_flat[i]
        
        disease_data = pbr_data[
            (pbr_data['Disease'] == disease) &
            (pbr_data['Corrected_PBR'] > 0)
        ].copy()
        
        # Plot base map with white background and thin light gray border
        world.plot(ax=ax, color='white', edgecolor='#cccccc', linewidth=0.3, alpha=1.0)
        
        if len(disease_data) > 0:
            world_disease = world.merge(
                disease_data[['ISO3', 'Corrected_log_PBR', 'Corrected_PBR']],
                left_on=iso_col,
                right_on='ISO3',
                how='left'
            )
            
            # Plot choropleth
            world_disease.plot(
                ax=ax,
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
            
            # Annotate top 4 over-represented countries (PBR > 3)
            over_rep = disease_data[disease_data['Corrected_PBR'] > 3].nlargest(4, 'Corrected_PBR')
            for _, country in over_rep.iterrows():
                iso3 = country['ISO3']
                country_geom = world[world[iso_col] == iso3]
                if not country_geom.empty:
                    try:
                        centroid = country_geom.geometry.centroid.iloc[0]
                        ax.annotate(
                            f"{iso3}\n{country['Corrected_log_PBR']:.2f}",
                            (centroid.x, centroid.y),
                            fontsize=6, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='darkred',
                                      alpha=0.25, edgecolor='red'),
                            color='black'
                        )
                    except:
                        pass
                        
            # Annotate top 4 under-represented countries (PBR < 0.5)
            under_rep = disease_data[
                (disease_data['Corrected_PBR'] > 0) &
                (disease_data['Corrected_PBR'] < 0.5)
            ].nsmallest(4, 'Corrected_PBR')
            for _, country in under_rep.iterrows():
                iso3 = country['ISO3']
                country_geom = world[world[iso_col] == iso3]
                if not country_geom.empty:
                    try:
                        centroid = country_geom.geometry.centroid.iloc[0]
                        ax.annotate(
                            f"{iso3}\n{country['Corrected_log_PBR']:.2f}",
                            (centroid.x, centroid.y),
                            fontsize=6, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='darkblue',
                                      alpha=0.25, edgecolor='blue'),
                            color='black'
                        )
                    except:
                        pass
        
        # Styles
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Format title
        p_total = cumulative_participants.get(disease, 0)
        c_count = len(disease_data)
        title_fmt = disease.replace(' and ', ' &\n') if len(disease) > 35 else disease
        ax.set_title(f'{title_fmt}\n({format_number(p_total)} participants, {c_count} countries)',
                     fontsize=10, fontweight='bold', pad=10)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    # Add horizontal colorbar
    cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.015])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap, norm=Normalize(vmin=-1.5, vmax=1.5))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\mathrm{Log}_{10}(\mathrm{PBR})$', fontsize=12, fontweight='bold', labelpad=10)
    
    # Legend panel
    legend_text = (
        "Participation-to-Burden Ratio (PBR) Interpretation:\n"
        "• Blue: Under-represented (log(PBR) < 0, fewer participants than burden warrants)\n"
        "• Light Gray: Proportional representation (log(PBR) ≈ 0)\n"
        "• Red: Over-represented (log(PBR) > 0, more participants than burden warrants)\n"
        "• White: No trial participant data\n"
        "• Dark red boxes: Examples of most over-represented countries (PBR > 3)\n"
        "• Dark blue boxes: Examples of most under-represented countries (PBR < 0.5)\n\n"
        f"Data range: 2000–2024, {pbr_data['ISO3'].nunique()} countries, "
        f"{pbr_data[pbr_data['Corrected_PBR'] > 0]['Total_Participants'].sum():,.0f} avg annual participants"
    )
    
    fig.text(0.02, 0.02, legend_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.7", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    fig.patch.set_facecolor('#fefcf6')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.95, bottom=0.15)
    
    png_out = os.path.join(output_dir, "extended_data_fig3.png")
    pdf_out = os.path.join(output_dir, "extended_data_fig3.pdf")
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight', facecolor='#fefcf6')
    plt.savefig(pdf_out, bbox_inches='tight', facecolor='#fefcf6')
    print(f"Figures successfully saved to:\n  - {png_out}\n  - {pdf_out}")

if __name__ == '__main__':
    generate_figure()
