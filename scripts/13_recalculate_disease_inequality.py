import os
import pandas as pd
import numpy as np
import geopandas as gpd

def calculate_inequality_table(n_countries=183):
    # Paths
    pmid_cause_path = '/Users/wen/Desktop/participation_inequality/CauseClassier/pmid_cause.csv'
    geoinfor_path = '/Users/wen/Desktop/participation_inequality/public/geoinfor183_disease_matched.csv'
    year_path = '/Users/wen/Desktop/participation_inequality/data/year_195k.csv'
    gbd_path = '/Users/wen/Desktop/participation_inequality/data/gbddisease.csv'
    c_map_path = '/Users/wen/Desktop/participation_inequality/data/country_mapping_for_figure.csv'
    app_path = '/Users/wen/Desktop/participation_inequality/public/APP_visual_factor_updated.csv'
    geojson_path = '/Users/wen/Desktop/participation_inequality/data/ne_110m_admin_0_countries.geojson'

    custom_diseases = [
        'Maternal and neonatal disorders',
        'Respiratory infections and tuberculosis',
        'Nutritional deficiencies',
        'HIV/AIDS and sexually transmitted infections',
        'Diabetes and kidney diseases',
        'Mental disorders',
        'Skin and subcutaneous diseases',
        'Chronic respiratory diseases',
        'Neglected tropical diseases and malaria',
        'Digestive diseases',
        'Cardiovascular diseases',
        'Neurological disorders',
        'Musculoskeletal disorders',
        'Substance use disorders',
        'Sense organ diseases',
        'Neoplasms'
    ]

    df_geo = pd.read_csv(geoinfor_path).rename(columns={'pmid': 'PMID', 'iso3': 'ISO3', 'amount': 'Amount'})
    df_year = pd.read_csv(year_path).rename(columns={'PMID': 'PMID', 'YEAR': 'YEAR'})
    df_cause = pd.read_csv(pmid_cause_path).rename(columns={'CAUSE': 'Cause'})
    df_gbd = pd.read_csv(gbd_path)
    app = pd.read_csv(app_path)
    world = gpd.read_file(geojson_path)

    if n_countries == 180:
        valid_iso3s = set(app['ISO3'].unique())
        df_geo = df_geo[df_geo['ISO3'].isin(valid_iso3s)]

    # Filter cause & year
    df_cause_l2 = df_cause[(df_cause['Level'] == 2) & (df_cause['Cause'].isin(custom_diseases))][['PMID', 'Cause']].drop_duplicates()
    df_year_filtered = df_year[(df_year['YEAR'] >= 2000) & (df_year['YEAR'] <= 2024)]

    study_data = df_cause_l2.merge(df_year_filtered, on='PMID', how='inner')
    participant_data = study_data.merge(df_geo, on='PMID', how='inner')

    # Income mapping (High Income = Global North)
    country_income = app[['ISO3', 'income_group']].drop_duplicates().set_index('ISO3')['income_group'].to_dict()

    # Continent mapping
    country_continent = world.set_index('ADM0_A3')['CONTINENT'].to_dict()
    iso_to_region = dict(zip(world['ADM0_A3'], world['REGION_UN']))

    participant_data['income_group'] = participant_data['ISO3'].map(country_income)
    participant_data['continent'] = participant_data['ISO3'].map(country_continent)
    participant_data['is_north'] = participant_data['income_group'] == 'H'
    participant_data['is_africa'] = (participant_data['continent'] == 'Africa') | (participant_data['ISO3'].map(iso_to_region) == 'Africa')

    # GBD DALY data
    gbd_filtered = df_gbd[
        (df_gbd['year'] >= 2000) &
        (df_gbd['year'] <= 2024) &
        (df_gbd['cause_name'].isin(custom_diseases))
    ].copy()

    gbd_to_iso = dict(zip(df_gbd['location_name'], df_gbd['ISO3']))
    gbd_filtered['ISO3'] = gbd_filtered['location_name'].map(gbd_to_iso)
    iso_set = set(df_geo['ISO3'].unique())
    gbd_filtered = gbd_filtered[gbd_filtered['ISO3'].isin(iso_set)]

    gbd_filtered['income_group'] = gbd_filtered['ISO3'].map(country_income)
    gbd_filtered['continent'] = gbd_filtered['ISO3'].map(country_continent)
    gbd_filtered['is_north'] = gbd_filtered['income_group'] == 'H'
    gbd_filtered['is_africa'] = (gbd_filtered['continent'] == 'Africa') | (gbd_filtered['ISO3'].map(iso_to_region) == 'Africa')

    # Average annual DALYs per country per disease
    daly_avg = gbd_filtered.groupby(['ISO3', 'cause_name', 'is_north', 'is_africa'])['val'].mean().reset_index()

    results = []

    for disease in custom_diseases:
        p_dis = participant_data[participant_data['Cause'] == disease]
        total_p = p_dis['Amount'].sum()
        north_p = p_dis[p_dis['is_north']]['Amount'].sum()
        africa_p = p_dis[p_dis['is_africa']]['Amount'].sum()

        d_dis = daly_avg[daly_avg['cause_name'] == disease]
        total_d = d_dis['val'].sum()
        north_d = d_dis[d_dis['is_north']]['val'].sum()
        south_d = d_dis[~d_dis['is_north']]['val'].sum()
        africa_d = d_dis[d_dis['is_africa']]['val'].sum()

        north_p_share = north_p / total_p * 100 if total_p > 0 else 0
        north_d_share = north_d / total_d * 100 if total_d > 0 else 0
        south_d_share = south_d / total_d * 100 if total_d > 0 else 0
        africa_d_share = africa_d / total_d * 100 if total_d > 0 else 0
        africa_p_share = africa_p / total_p * 100 if total_p > 0 else 0

        ratio = north_p_share / north_d_share if north_d_share > 0 else 0

        results.append({
            'Disease Category': disease,
            'Total Participants': f'{total_p/1e6:.2f}M',
            'Global North Participant Share': f'{north_p_share:.2f}%',
            'Global North DALY Share (Burden)': f'{north_d_share:.2f}%',
            'Global South DALY Share (Burden)': f'{south_d_share:.2f}%',
            'Africa DALY Share (Burden)': f'{africa_d_share:.2f}%',
            'Africa Participant Share': f'{africa_p_share:.2f}%',
            'Inequality Ratio (North Part% / North DALY%)': f'{ratio:.2f}x',
            'ratio_raw': ratio
        })

    res_df = pd.DataFrame(results).sort_values(by='ratio_raw', ascending=False).reset_index(drop=True)
    res_df['Rank'] = res_df.index + 1
    cols = [
        'Rank', 'Disease Category', 'Total Participants',
        'Global North Participant Share', 'Global North DALY Share (Burden)',
        'Global South DALY Share (Burden)', 'Africa DALY Share (Burden)',
        'Africa Participant Share', 'Inequality Ratio (North Part% / North DALY%)'
    ]
    return res_df[cols]

def df_to_markdown_table(df):
    headers = list(df.columns)
    header_line = '| ' + ' | '.join(headers) + ' |'
    sep_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    rows = []
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join([str(val) for val in row]) + ' |')
    return '\n'.join([header_line, sep_line] + rows)

if __name__ == '__main__':
    df183 = calculate_inequality_table(183)
    df180 = calculate_inequality_table(180)

    md_out = '/Users/wen/Desktop/participation_inequality/public/disease_inequality_table.md'
    
    with open(md_out, 'w') as f:
        f.write('# Recalculated Disease Inequality Table\n\n')
        f.write('## Harmonized Primary Sample: N = 183 Countries (Disease & Clinical Trial Matched)\n\n')
        f.write('This table presents the global clinical trial participant distribution vs. GBD DALY burden across 16 major disease categories, strictly calculated on the **183 disease-matched country sample**.\n\n')
        f.write(df_to_markdown_table(df183))
        f.write('\n\n---\n\n')
        f.write('## Sensitivity Sample: N = 180 Countries (Econometric Sample with Complete Macro Indicators)\n\n')
        f.write('This table presents the same calculation restricted to the **180 macroeconomic indicator country sample**.\n\n')
        f.write(df_to_markdown_table(df180))

    print(f'Successfully calculated and saved disease inequality tables to:\n  - {md_out}')
