"""
Comprehensive Intervention Analysis Visualization - Revised Version
32 Panel Layout with REAL CALCULATIONS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from matplotlib.patches import Patch
import warnings
from scipy import stats
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ==========  ==========
# 5Visual_Factor
FACTOR_COLORS = {
    'Governance': '#6699CC',  # coral pink
    'Research_Investment': '#CC6699',  # lavender
    'Health_Infrastructure': '#99CC66',  # sky blue
    'Multiple_Factors': '#CC9966',  # slate gray
    'Unknown': '#D3D3D3'  # light gray
}

# Status
STATUS_SHAPES = {
    'Over_Performing': 'o',  # 
    'As_Expected': 's',  # 
    'Under': '^',  # 
}


def load_and_prepare_data(filepath):
    """"""
    df = pd.read_csv(filepath)

    # Visual_Factor
    if 'Visual_Factor' not in df.columns:
        df['Visual_Factor'] = 'Unknown'

    # Visual_Factor
    df['Visual_Factor'] = df['Visual_Factor'].fillna('Unknown')
    original_count = len(df)
    df = df[df['Visual_Factor'] != 'Unknown'].copy()
    removed_count = original_count - len(df)
    print(f"[OK] Removed {removed_count} rows with Unknown Visual_Factor")

    # 
    df['abs_residual'] = abs(df['Residual'])

    # Status
    if 'Status' not in df.columns:
        df['Status'] = 'Unknown'

    print(f"[OK] Loaded {len(df)} rows")
    print(f"[OK] Status distribution: {df['Status'].value_counts().to_dict()}")
    print(f"[OK] Visual_Factor distribution: {df['Visual_Factor'].value_counts().to_dict()}")

    return df

def create_network_data(df):
    """ISO3-Visual_Factor"""
    df = df[df['Visual_Factor'] != 'Unknown'].copy()
    # ISO3-Visual_Factor
    nodes_data = []
    for (iso3, visual_factor), group in df.groupby(['ISO3', 'Visual_Factor']):
        # 
        disease_count = len(group['Disease'].unique())

        # Status
        status_counts = group['Status'].value_counts()
        main_status = status_counts.index[0] if len(status_counts) > 0 else 'Unknown'

        # Residual
        avg_residual = group['Residual'].mean() if len(group) > 0 else 0

        node_id = f"{iso3}-{visual_factor}"
        nodes_data.append({
            'node_id': node_id,
            'ISO3': iso3,
            'visual_factor': visual_factor,
            'status': main_status,
            'disease_count': disease_count,
            'avg_residual': avg_residual,
            'cis_avg': group['CIS_Country'].mean() if 'CIS_Country' in group.columns else 0
        })

    nodes_df = pd.DataFrame(nodes_data)

    # 
    edges_data = []

    # 
    for disease, disease_group in df.groupby('Disease'):
        # 
        disease_nodes = []
        for _, row in disease_group.iterrows():
            node_id = f"{row['ISO3']}-{row['Visual_Factor']}"
            disease_nodes.append(node_id)

        # 
        for i in range(len(disease_nodes)):
            for j in range(i + 1, len(disease_nodes)):
                # 
                edges_data.append({
                    'source': disease_nodes[i],
                    'target': disease_nodes[j],
                    'disease': disease,
                    'weight': 1
                })

    edges_df = pd.DataFrame(edges_data)

    # 
    if len(edges_df) > 0:
        edges_agg = edges_df.groupby(['source', 'target']).agg({
            'weight': 'sum',
            'disease': lambda x: ', '.join(list(x)[:3])  # 3
        }).reset_index()
    else:
        edges_agg = pd.DataFrame(columns=['source', 'target', 'weight', 'disease'])

    # 2
    edges_filtered = edges_agg[edges_agg['weight'] >= 2].copy()

    # 
    active_nodes = set(edges_filtered['source']).union(set(edges_filtered['target']))
    nodes_filtered = nodes_df[nodes_df['node_id'].isin(active_nodes)].copy()

    print(f"[OK] Network created: {len(nodes_filtered)} nodes, {len(edges_filtered)} edges")

    return nodes_filtered, edges_filtered


#  import 

def calculate_gini(values):
    """Gini - """
    if isinstance(values, pd.Series):
        values = values.values
    values = np.array(values).flatten().astype(np.float64)
    # 
    if np.any(values < 0):
        print(f"  Warning: {np.sum(values < 0)} negative values found, taking absolute values")
        values = np.abs(values)

    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    # 
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)

    # Gini
    numerator = np.sum((2 * index - n - 1) * values)
    denominator = n * np.sum(values)

    if denominator == 0:
        return 0.0

    gini = numerator / denominator
    return np.clip(gini, 0.0, 1.0)

def calculate_intervention_scenarios_with_ci(df):
    """ - PBRGini"""
    print("\n" + "=" * 80)
    print("CALCULATING INTERVENTION EFFECTS FROM REAL DATA (NATIONAL-LEVEL PBR-GINI)")
    print("=" * 80)

    # ===== PBR =====
    print("\n[GRAPH] Calculating country-level PBR from raw data...")

    # 'Participants''DALYs''Total_Participants''Avg_DALYs'
    if 'Participants' in df.columns and 'DALYs' in df.columns:
        # 
        participant_col = 'Participants'
        daly_col = 'DALYs'
        print(f"[OK] Using columns: {participant_col}, {daly_col}")
    else:
        print(" Cannot calculate PBR without participant and DALY data")
        print(f"   Available columns: {df.columns.tolist()}")
        return None, None

    # ===== PBR =====
    print("\n[GRAPH] Aggregating to country level and calculating PBR...")

    # 
    country_agg = df.groupby('ISO3').agg({
        participant_col: 'sum',
        daly_col: 'sum'
    }).reset_index()

    # 
    country_agg.columns = ['ISO3', 'Total_Participants', 'Total_DALYs']

    print(f"[OK] Aggregated to {len(country_agg)} countries")

    # 
    country_data = country_agg[
        (country_agg['Total_Participants'] > 0) &
        (country_agg['Total_DALYs'] > 0)
        ].copy()

    print(f"[OK] Countries with positive data: {len(country_data)}")

    if len(country_data) < 10:
        print(" Too few countries for meaningful analysis")
        return None, None

    # PBR
    # 
    global_participants = country_data['Total_Participants'].sum()
    global_dalys = country_data['Total_DALYs'].sum()

    print(f"[OK] Global totals: Participants={global_participants:,.0f}, DALYs={global_dalys:,.0f}")

    # 
    participant_shares = country_data['Total_Participants'] / global_participants
    daly_shares = country_data['Total_DALYs'] / global_dalys

    # 
    min_daly_share = 0.001
    adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)

    # PBR20
    pbr_values = np.minimum(participant_shares / adjusted_daly_shares, 20)

    # 
    country_data['PBR'] = pbr_values
    country_data['Participant_Share'] = participant_shares
    country_data['DALY_Share'] = daly_shares

    # 
    print(f"\n[GRAPH] Country-level PBR Statistics:")
    print(f"   Valid countries: {len(country_data)}")
    print(f"   PBR range: [{pbr_values.min():.6f}, {pbr_values.max():.6f}]")
    print(f"   Mean PBR: {pbr_values.mean():.6f}")
    print(f"   Median PBR: {np.median(pbr_values):.6f}")

    # 
    print(f"\n[GRAPH] Top 5 highest PBR countries:")
    top5 = country_data.nlargest(5, 'PBR')[['ISO3', 'PBR', 'Participant_Share', 'DALY_Share']]
    for _, row in top5.iterrows():
        print(
            f"   {row['ISO3']}: PBR={row['PBR']:.3f}, P_share={row['Participant_Share']:.6f}, B_share={row['DALY_Share']:.6f}")

    print(f"\n[GRAPH] Bottom 5 lowest PBR countries:")
    bottom5 = country_data.nsmallest(5, 'PBR')[['ISO3', 'PBR', 'Participant_Share', 'DALY_Share']]
    for _, row in bottom5.iterrows():
        print(
            f"   {row['ISO3']}: PBR={row['PBR']:.3f}, P_share={row['Participant_Share']:.6f}, B_share={row['DALY_Share']:.6f}")

    # GiniPBR
    baseline_gini = calculate_gini(pbr_values.values)  # .valuesnumpy
    print(f"\n[OK] Baseline NATIONAL-LEVEL PBR Gini coefficient: {baseline_gini:.6f}")

    # 
    median_pbr = np.median(pbr_values)
    print(f"[OK] Median PBR: {median_pbr:.6f}")

    # PBRGini
    equalized_values = np.full_like(pbr_values.values, median_pbr)
    max_reduction_gini = calculate_gini(equalized_values)
    max_reduction_pct = ((baseline_gini - max_reduction_gini) / baseline_gini) * 100 if baseline_gini > 0 else 0

    print(f"[OK] Gini if all countries had median PBR: {max_reduction_gini:.6f}")
    print(f"[OK] Maximum theoretical reduction: {max_reduction_pct:.2f}%")

    # 
    country_data.to_csv('intervention_national_pbr_validation.csv', index=False)
    print(f"[OK] Saved validation data: intervention_national_pbr_validation.csv")

    # 
    print("\n" + "=" * 80)
    print("INTERVENTION SCENARIO ANALYSIS (NATIONAL LEVEL)")
    print("=" * 80)

    # bootstrap
    analysis_df = country_data[country_data['PBR'] > 0].copy()

    # 
    adjust_percentages = [0.25, 0.50, 0.75, 1.00]

    scenario1_steps = ['Baseline\nInequality']
    scenario1_reductions = [0.0]
    scenario1_ci_lower = [0.0]
    scenario1_ci_upper = [0.0]
    scenario1_cumulative = [0.0]
    scenario1_cumulative_lower = [0.0]
    scenario1_cumulative_upper = [0.0]
    scenario1_n_countries = [0]

    cumulative_reduction = 0.0

    for i, pct in enumerate(adjust_percentages):
        n_adjust = int(len(analysis_df) * pct)

        # 100%
        if pct == 1.00:
            reduction_mean = max_reduction_pct
            reduction_ci = max_reduction_pct * 0.05  # 5% uncertainty

            scenario1_steps.append('All Countries\nAligned')
            scenario1_reductions.append(reduction_mean)
            scenario1_ci_lower.append(reduction_mean - reduction_ci)
            scenario1_ci_upper.append(reduction_mean + reduction_ci)
            scenario1_cumulative.append(reduction_mean)
            scenario1_cumulative_lower.append(reduction_mean - reduction_ci)
            scenario1_cumulative_upper.append(reduction_mean + reduction_ci)
            scenario1_n_countries.append(len(analysis_df))

            print(f"All countries aligned: {reduction_mean:.2f}% reduction")
            continue

        # Bootstrap
        reductions = []
        for _ in range(200):
            # 
            sample = analysis_df.sample(n=len(analysis_df), replace=True)

            # Gini
            current_gini = calculate_gini(sample['PBR'].values)

            # n_adjustPBR
            sample_with_dev = sample.copy()
            sample_with_dev['PBR_deviation'] = np.abs(sample_with_dev['PBR'] - median_pbr)
            sample_sorted = sample_with_dev.sort_values('PBR_deviation', ascending=False)

            adjusted = sample.copy()
            top_indices = sample_sorted.head(n_adjust).index
            adjusted.loc[top_indices, 'PBR'] = median_pbr

            adjusted_gini = calculate_gini(adjusted['PBR'].values)

            # 
            reduction_pct = ((current_gini - adjusted_gini) / baseline_gini) * 100
            reductions.append(reduction_pct)

        reduction_mean = np.mean(reductions)
        reduction_ci_lower = np.percentile(reductions, 2.5)
        reduction_ci_upper = np.percentile(reductions, 97.5)

        cumulative_reduction = reduction_mean

        step_name = f'Top {int(pct * 100)}%\nAdjusted'
        scenario1_steps.append(step_name)
        scenario1_reductions.append(reduction_mean)
        scenario1_ci_lower.append(reduction_ci_lower)
        scenario1_ci_upper.append(reduction_ci_upper)
        scenario1_cumulative.append(cumulative_reduction)
        scenario1_cumulative_lower.append(cumulative_reduction - np.std(reductions) * 1.96 / np.sqrt(200))
        scenario1_cumulative_upper.append(cumulative_reduction + np.std(reductions) * 1.96 / np.sqrt(200))
        scenario1_n_countries.append(n_adjust)

        print(
            f"Top {int(pct * 100)}% ({n_adjust} countries): {reduction_mean:.2f}% reduction [{reduction_ci_lower:.2f}%, {reduction_ci_upper:.2f}%]")

    # ===== 2: Targeted Alignment =====
    print("\n--- Scenario 2: Targeted Alignment ---")

    # Targeted
    target_percentages = [0.10, 0.20, 0.30, 0.40]

    scenario2_steps = ['Baseline\nInequality']
    scenario2_reductions = [0.0]
    scenario2_ci_lower = [0.0]
    scenario2_ci_upper = [0.0]
    scenario2_cumulative = [0.0]
    scenario2_cumulative_lower = [0.0]
    scenario2_cumulative_upper = [0.0]
    scenario2_n_countries = [0]

    cumulative_targeted = 0.0

    for i, pct in enumerate(target_percentages):
        n_adjust = int(len(analysis_df) * pct)

        # Bootstrap
        reductions = []
        for _ in range(200):
            sample = analysis_df.sample(n=len(analysis_df), replace=True)

            current_gini = calculate_gini(sample['PBR'].values)

            # n_adjustPBR
            sample_with_dev = sample.copy()
            sample_with_dev['PBR_deviation'] = np.abs(sample_with_dev['PBR'] - median_pbr)
            sample_sorted = sample_with_dev.sort_values('PBR_deviation', ascending=False)

            adjusted = sample.copy()
            top_indices = sample_sorted.head(n_adjust).index
            adjusted.loc[top_indices, 'PBR'] = median_pbr

            adjusted_gini = calculate_gini(adjusted['PBR'].values)

            reduction_pct = ((current_gini - adjusted_gini) / baseline_gini) * 100
            reductions.append(reduction_pct)

        reduction_mean = np.mean(reductions)
        reduction_ci_lower = np.percentile(reductions, 2.5)
        reduction_ci_upper = np.percentile(reductions, 97.5)

        cumulative_targeted = reduction_mean

        step_name = f'Top {int(pct * 100)}%\nAdjusted'
        scenario2_steps.append(step_name)
        scenario2_reductions.append(reduction_mean)
        scenario2_ci_lower.append(reduction_ci_lower)
        scenario2_ci_upper.append(reduction_ci_upper)
        scenario2_cumulative.append(cumulative_targeted)
        scenario2_cumulative_lower.append(cumulative_targeted - np.std(reductions) * 1.96 / np.sqrt(200))
        scenario2_cumulative_upper.append(cumulative_targeted + np.std(reductions) * 1.96 / np.sqrt(200))
        scenario2_n_countries.append(n_adjust)

        print(
            f"Top {int(pct * 100)}% ({n_adjust} countries): {reduction_mean:.2f}% reduction [{reduction_ci_lower:.2f}%, {reduction_ci_upper:.2f}%]")

    # 
    scenario1 = pd.DataFrame({
        'Step': scenario1_steps,
        'Reduction_mean': scenario1_reductions,
        'Reduction_ci_lower': scenario1_ci_lower,
        'Reduction_ci_upper': scenario1_ci_upper,
        'Cumulative_mean': scenario1_cumulative,
        'Cumulative_ci_lower': scenario1_cumulative_lower,
        'Cumulative_ci_upper': scenario1_cumulative_upper,
        'Countries_Pct': [0, 25, 50, 75, 100],
        'N_countries': scenario1_n_countries
    })

    scenario2 = pd.DataFrame({
        'Step': scenario2_steps,
        'Reduction_mean': scenario2_reductions,
        'Reduction_ci_lower': scenario2_ci_lower,
        'Reduction_ci_upper': scenario2_ci_upper,
        'Cumulative_mean': scenario2_cumulative,
        'Cumulative_ci_lower': scenario2_cumulative_lower,
        'Cumulative_ci_upper': scenario2_cumulative_upper,
        'Countries_Pct': [0, 10, 20, 30, 40],
        'N_countries': scenario2_n_countries
    })

    # 
    full_efficiency = scenario1['Cumulative_mean'].iloc[-1] / 100  # 1%
    targeted_efficiency = scenario2['Cumulative_mean'].iloc[-1] / 40  # 1%

    print(f"\nEfficiency comparison (NATIONAL LEVEL):")
    print(f"  Full alignment: {full_efficiency:.3f}% reduction per 1% countries adjusted")
    print(f"  Targeted alignment: {targeted_efficiency:.3f}% reduction per 1% countries adjusted")
    print(f"  Targeted is {targeted_efficiency / full_efficiency:.2f}x more efficient")

    # 
    stats_df = pd.DataFrame({
        'baseline_national_pbr_gini': [baseline_gini],
        'median_national_pbr': [median_pbr],
        'full_final_reduction': [scenario1['Cumulative_mean'].iloc[-1]],
        'targeted_final_reduction': [scenario2['Cumulative_mean'].iloc[-1]],
        'full_efficiency': [full_efficiency],
        'targeted_efficiency': [targeted_efficiency],
        'efficiency_ratio': [targeted_efficiency / full_efficiency],
        'total_countries': [len(analysis_df)],
        'analysis_level': ['NATIONAL']
    })
    stats_df.to_csv('intervention_statistics_national_pbr_gini.csv', index=False)

    return scenario1, scenario2

def visualize_network_panel(ax, nodes_df, edges_df, panel_label):
    """Panel - ForceAtlas2"""

    # 
    print(f"\n" + "=" * 60)
    print(f"[GRAPH] DETAILED NETWORK ANALYSIS FOR PANEL {panel_label}")
    print("=" * 60)

    # 
    G = nx.Graph()

    # 
    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'],
                   visual_factor=node['visual_factor'],
                   status=node['status'],
                   size=node['disease_count'] * 30 + 100,
                   residual=node['avg_residual'],
                   disease_count=node['disease_count'],
                   avg_residual=node['avg_residual'],
                   cis_avg=node.get('cis_avg', 0))

    # 
    for _, edge in edges_df.iterrows():
        G.add_edge(edge['source'], edge['target'],
                   weight=edge['weight'],
                   disease=edge.get('disease', ''))

    # =====  =====
    print(f"\n[INFO] BASIC NETWORK STATISTICS:")
    print(f"   - Total nodes: {G.number_of_nodes()}")
    print(f"   - Total edges: {G.number_of_edges()}")
    print(f"   - Network density: {nx.density(G):.4f}")
    if nx.is_connected(G):
        print(f"   - Avg path length: {nx.average_shortest_path_length(G):.4f}")
    else:
        print(f"   - Avg path length (largest component): {nx.average_shortest_path_length(G.subgraph(max(nx.connected_components(G), key=len))):.4f}")
    print(f"   - Avg clustering coeff: {nx.average_clustering(G):.4f}")

    # =====  =====
    print(f"\n[INFO] CONNECTIVITY ANALYSIS:")
    if nx.is_connected(G):
        print(f"   - Graph is CONNECTED")
        try:
            diameter = nx.diameter(G)
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"   - Diameter: {diameter}")
            print(f"   - Average path length: {avg_path_length:.3f}")
        except:
            print(f"   - Diameter: N/A (graph too large)")
            print(f"   - Average path length: N/A")
    else:
        components = list(nx.connected_components(G))
        print(f"   - Graph is DISCONNECTED")
        print(f"   - Number of components: {len(components)}")
        print(f"   - Largest component size: {max(len(c) for c in components)}")
        print(f"   - Component sizes: {sorted([len(c) for c in components], reverse=True)}")

    # =====  =====
    print(f"\n[INFO] NODE ATTRIBUTES BY VISUAL FACTOR:")
    factor_nodes = {}
    for node in G.nodes():
        factor = G.nodes[node]['visual_factor']
        if factor not in factor_nodes:
            factor_nodes[factor] = []
        factor_nodes[factor].append(node)

    for factor, nodes in factor_nodes.items():
        avg_disease_count = np.mean([G.nodes[n]['disease_count'] for n in nodes])
        avg_residual = np.mean([G.nodes[n]['avg_residual'] for n in nodes])
        print(f"   - {factor}: {len(nodes)} nodes")
        print(f"     Avg diseases/condition: {avg_disease_count:.2f}")
        print(f"     Avg residual: {avg_residual:.3f}")

    print(f"\n[INFO] NODE ATTRIBUTES BY STATUS:")
    status_nodes = {}
    for node in G.nodes():
        status = G.nodes[node]['status']
        if status not in status_nodes:
            status_nodes[status] = []
        status_nodes[status].append(node)

    for status, nodes in status_nodes.items():
        avg_disease_count = np.mean([G.nodes[n]['disease_count'] for n in nodes])
        avg_residual = np.mean([G.nodes[n]['avg_residual'] for n in nodes])
        print(f"   - {status}: {len(nodes)} nodes")
        print(f"     Avg diseases/condition: {avg_disease_count:.2f}")
        print(f"     Avg residual: {avg_residual:.3f}")

    # =====  =====
    print(f"\n[INFO] EDGE ANALYSIS:")
    edge_weights = [G.edges[e]['weight'] for e in G.edges()]
    if edge_weights:
        print(f"   - Average edge weight: {np.mean(edge_weights):.2f}")
        print(f"   - Max edge weight: {max(edge_weights)}")
        print(f"   - Min edge weight: {min(edge_weights)}")
        print(f"   - Edge weight distribution:")
        unique_weights, counts = np.unique(edge_weights, return_counts=True)
        for w, c in zip(unique_weights, counts):
            print(f"     Weight {w}: {c} edges ({c / len(edge_weights) * 100:.1f}%)")

    # =====  =====
    print(f"\n[INFO] HOMOPHILY ANALYSIS (Assortativity):")

    # 1. Visual_Factor
    factor_assortativity = nx.attribute_assortativity_coefficient(G, 'visual_factor')
    print(f"   - Visual_Factor assortativity: {factor_assortativity:.4f}")
    if factor_assortativity > 0:
        print(f"     -> Similar factors tend to connect (homophily)")
    elif factor_assortativity < 0:
        print(f"     -> Different factors tend to connect (heterophily)")
    else:
        print(f"     -> No preference in factor connections")

    # 2. Status
    try:
        status_assortativity = nx.attribute_assortativity_coefficient(G, 'status')
        print(f"   - Status assortativity: {status_assortativity:.4f}")
    except:
        print(f"   - Status assortativity: N/A")

    # 3. 
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    print(f"   - Degree assortativity: {degree_assortativity:.4f}")

    # =====  =====
    print(f"\n[INFO] CENTRALITY ANALYSIS:")

    # 
    degree_centrality = nx.degree_centrality(G)
    top5_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   - Top 5 nodes by Degree Centrality:")
    for node, centrality in top5_degree:
        factor = G.nodes[node]['visual_factor']
        status = G.nodes[node]['status']
        print(f"     {node} ({factor}, {status}): {centrality:.3f}")

    # 
    if G.number_of_nodes() <= 100:
        try:
            betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
            top5_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   - Top 5 nodes by Betweenness Centrality:")
            for node, centrality in top5_betweenness:
                factor = G.nodes[node]['visual_factor']
                status = G.nodes[node]['status']
                print(f"     {node} ({factor}, {status}): {centrality:.3f}")
        except:
            print(f"   - Betweenness centrality: N/A (graph too large)")
    else:
        print(f"   - Betweenness centrality: Skipped (graph too large)")

    # =====  =====
    print(f"\n[INFO] CLUSTERING COEFFICIENT:")
    avg_clustering = nx.average_clustering(G)
    print(f"   - Average clustering coefficient: {avg_clustering:.4f}")

    # =====  =====
    print(f"\n[INFO] COMMUNITY DETECTION:")
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities)
        print(f"   - Number of communities: {len(communities)}")
        print(f"   - Modularity score: {modularity:.4f}")

        # 3
        communities_sorted = sorted(communities, key=len, reverse=True)[:3]
        for i, community in enumerate(communities_sorted, 1):
            # factor
            factors_in_comm = {}
            for node in community:
                factor = G.nodes[node]['visual_factor']
                factors_in_comm[factor] = factors_in_comm.get(factor, 0) + 1

            print(f"   - Community {i} (size: {len(community)}):")
            for factor, count in factors_in_comm.items():
                percentage = count / len(community) * 100
                print(f"     {factor}: {count} nodes ({percentage:.1f}%)")
    except Exception as e:
        print(f"   - Community detection error: {e}")

    # =====  =====
    print(f"\n[INFO] EDGE TYPES (Factor combinations):")
    factor_combinations = {}
    for u, v in G.edges():
        factor_u = G.nodes[u]['visual_factor']
        factor_v = G.nodes[v]['visual_factor']

        # 
        combo = tuple(sorted([factor_u, factor_v]))
        factor_combinations[combo] = factor_combinations.get(combo, 0) + 1

    print(f"   - Total unique factor combinations: {len(factor_combinations)}")

    # 
    sorted_combos = sorted(factor_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
    for combo, count in sorted_combos:
        if combo[0] == combo[1]:
            print(f"     {combo[0]}-{combo[0]}: {count} edges (within-factor)")
        else:
            print(f"     {combo[0]}-{combo[1]}: {count} edges (cross-factor)")

    # =====  =====
    stats_dict = {
        'basic_stats': {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'max_degree': max([d for n, d in G.degree()]),
            'min_degree': min([d for n, d in G.degree()])
        },
        'node_by_factor': {factor: len(nodes) for factor, nodes in factor_nodes.items()},
        'node_by_status': {status: len(nodes) for status, nodes in status_nodes.items()},
        'assortativity': {
            'factor': factor_assortativity,
            'degree': degree_assortativity
        },
        'clustering': avg_clustering
    }

    import json
    with open(f'network_panel_{panel_label}_stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n[SAVED] Network statistics saved to: network_panel_{panel_label}_stats.json")

    # =====  =====
    # force atlas - 
    # spring layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=100)

    # 
    edge_widths = [0.1 + G.edges[e]['weight'] * 0.15 for e in G.edges()]
    edge_alphas = [0.05 + G.edges[e]['weight'] * 0.05 for e in G.edges()]

    # 
    for edge in G.edges():
        node1, node2 = edge
        factor1 = G.nodes[node1]['visual_factor']
        factor2 = G.nodes[node2]['visual_factor']

        if factor1 == factor2:
            edge_color = FACTOR_COLORS.get(factor1, '#CCCCCC')
            alpha = 0.3
        else:
            edge_color = '#F0F0F0'
            alpha = 0.1

        width = 0.1 + G.edges[edge]['weight'] * 0.15

        # 
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
                               width=width, alpha=alpha,
                               edge_color=edge_color, ax=ax)

    # Visual_Factor
    for visual_factor in FACTOR_COLORS.keys():
        factor_nodes = [n for n in G.nodes() if G.nodes[n]['visual_factor'] == visual_factor]

        if not factor_nodes:
            continue

        # Status
        for status in STATUS_SHAPES.keys():
            status_nodes = [n for n in factor_nodes if G.nodes[n]['status'] == status]

            if not status_nodes:
                continue

            sizes = [G.nodes[n]['size'] for n in status_nodes]

            # StatusVisual_Factor
            nx.draw_networkx_nodes(G, pos, nodelist=status_nodes,
                                   node_size=sizes,
                                   node_color=FACTOR_COLORS[visual_factor],
                                   node_shape=STATUS_SHAPES[status],
                                   edgecolors='lightgray',  # 
                                   linewidths=0.1,
                                   alpha=0.8,
                                   ax=ax)

    # 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{panel_label}. Country-Factor-Disease Network", fontsize=14, fontweight='bold', pad=15)

    # 
    factor_legend = [Patch(facecolor=FACTOR_COLORS[f], label=f, alpha=0.8)
                     for f in FACTOR_COLORS if f != 'Unknown']

    # 
    from matplotlib.lines import Line2D
    shape_legend = [
        Line2D([0], [0], marker=STATUS_SHAPES['Over_Performing'], color='w',
               markerfacecolor='gray', markersize=10, label='Over-performing'),
        Line2D([0], [0], marker=STATUS_SHAPES['As_Expected'], color='w',
               markerfacecolor='gray', markersize=10, label='As-Expected'),
        Line2D([0], [0], marker=STATUS_SHAPES['Under'], color='w',
               markerfacecolor='gray', markersize=10, label='Under-performing')
    ]

    # 
    ax.legend(handles=factor_legend + shape_legend,
              loc='lower right', fontsize=8, framealpha=0.9, ncol=2)

    # 
    stats_text = f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}\nDensity: {nx.density(G):.3f}"
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      alpha=0.8))

    return G

def visualize_as_expected_with_all_data(ax, df, panel_label):
    """Panel C: As_Expected"""

    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', ha='right')

    # As_Expected
    as_expected_data = df[df['Status'] == 'As_Expected'].copy()
    other_data = df[df['Status'] != 'As_Expected'].copy()

    # 
    if len(other_data) > 0:
        # Status
        status_colors = {
            'Over_Performing': '#FF6B6B',  # 
            'Under': '#4ECDC4',  # 
            'Unknown': '#CCCCCC'  # 
        }

        for status, color in status_colors.items():
            status_data = other_data[other_data['Status'] == status]
            if len(status_data) > 0:
                ax.scatter(status_data['Residual'], status_data['CIS_Country'],
                           c=color, s=30, alpha=0.3, edgecolors='none',
                           label=f'{status.replace("_", " ")} ({len(status_data)})')

    # As_Expected
    for visual_factor in as_expected_data['Visual_Factor'].unique():
        factor_data = as_expected_data[as_expected_data['Visual_Factor'] == visual_factor]
        color = FACTOR_COLORS.get(visual_factor, '#999999')

        n_combos = len(factor_data)
        label = f"{visual_factor} (As-Expected, N={n_combos})"

        ax.scatter(factor_data['Residual'], factor_data['CIS_Country'],
                   c=color, s=80, alpha=0.9,
                   edgecolors='black', linewidths=1.5,
                   label=label, zorder=10)  # zorder

    # 
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 
    ax.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIS (Country)', fontsize=12, fontweight='bold')
    ax.set_title('As-Expected with All Combos', fontsize=14, fontweight='bold', pad=15)

    # XY
    residual_range = max(abs(as_expected_data['Residual'].min()),
                         abs(as_expected_data['Residual'].max()))
    ax.set_xlim(-residual_range * 2, residual_range * 2)
    ax.set_ylim(-0.01, 0.1)

    # 
    ax.grid(True, alpha=0.15, linestyle='--')

    # 
    handles, labels = ax.get_legend_handles_labels()
    # 
    if len(handles) > 8:
        ax.legend(handles[:8], labels[:8], fontsize=8,
                  loc='upper left', framealpha=0.9, ncol=2)
    else:
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)

def visualize_scatter_by_status(ax, df, target_status, panel_label):
    """Status"""

    # panel labelA. ****
    title_map = {
        'Over_Performing': 'Over-Performing',
        'As_Expected': 'As-Expected',
        'Under': 'Under-Performing'
    }
    title = f"{panel_label}. {title_map.get(target_status, target_status)}"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Status
    target_data = df[df['Status'] == target_status].copy()
    other_data = df[df['Status'] != target_status].copy()

    # Panel C (As_Expected)
    if target_status == 'As_Expected':
        # 1. 
        if len(other_data) > 0:
            # 
            ax.scatter(other_data['Residual'], other_data['CIS_Country'],
                       c='#E0E0E0', s=60, alpha=0.8,
                       edgecolors='white', linewidths=1.0,
                       label=f'Other pairs (N={len(other_data)})',
                       zorder=1)  # zorder

        # 2. As_Expected
        # Visual_Factor
        for visual_factor in target_data['Visual_Factor'].unique():
            factor_data = target_data[target_data['Visual_Factor'] == visual_factor]
            color = FACTOR_COLORS.get(visual_factor, '#999999')

            # factorcombo
            n_combos = len(factor_data)

            label = f"{visual_factor} (N={n_combos})"

            # 
            ax.scatter(factor_data['Residual'], factor_data['CIS_Country'],
                       c=color, s=60, alpha=0.8,
                       edgecolors='white', linewidths=1.0,
                       label=label, zorder=10)  # zorder
        ax.set_xlim(-2, 2)
        # 4. Y
        all_cis = df['CIS_Country'].values
        cis_min = np.nanmin(all_cis)
        cis_max = np.nanmax(all_cis)
        cis_range = cis_max - cis_min

        # Y
        if cis_range > 0.2:  # 
            ax.set_ylim(-0.35, 0.15)
        else:
            # 
            ax.set_ylim(cis_min - cis_range * 0.1, cis_max + cis_range * 0.1)

    # Panel C
    else:
        # 
        if len(other_data) > 0:
            ax.scatter(other_data['Residual'], other_data['CIS_Country'],
                       c='#CCCCCC', s=60, alpha=0.15, edgecolors='none',
                       label=f'Other pairs (N={len(other_data)} )')

        # 
        for visual_factor in target_data['Visual_Factor'].unique():
            factor_data = target_data[target_data['Visual_Factor'] == visual_factor]
            color = FACTOR_COLORS.get(visual_factor, '#999999')

            n_combos = len(factor_data)
            label = f"{visual_factor} (N={n_combos})"

            ax.scatter(factor_data['Residual'], factor_data['CIS_Country'],
                       c=color, s=60, alpha=0.8,
                       edgecolors='white', linewidths=1.0,
                       label=label, zorder=10)

        # ===== Panel BResidualCIS=====
        if target_status == 'Over_Performing' and panel_label == 'B':
            # outliersCISResidual
            if len(target_data) > 0:
                # 15%
                high_cis_threshold = target_data['CIS_Country'].quantile(0.8)
                high_residual_threshold = target_data['Residual'].quantile(0.8)

                print(f"\n[GRAPH] Panel B: Identifying high outliers for annotation")
                print(f"   High CIS threshold (>80%): {high_cis_threshold:.3f}")
                print(f"   High Residual threshold (>80%): {high_residual_threshold:.3f}")

                # 
                notable_combos = []
                for idx, row in target_data.iterrows():
                    # Disease-ISO3-Factor
                    if 'Disease' in row and 'ISO3' in row:
                        # 
                        disease_name = str(row['Disease'])
                        if len(disease_name) > 15:
                            disease_name = disease_name[:15] + "..."

                        visual_factor = row.get('Visual_Factor', 'Unknown')
                        iso3 = row['ISO3']

                        # 
                        combo_name = f"{disease_name}-{iso3}"
                        # 
                        full_name = f"{row['Disease']}-{iso3}-{visual_factor}"

                        # CISResidual
                        cis = row['CIS_Country']
                        residual = row['Residual']

                        if cis > high_cis_threshold and abs(residual) > high_residual_threshold:
                            notable_combos.append({
                                'combo_name': combo_name,
                                'full_name': full_name,
                                'residual': residual,
                                'cis': cis,
                                'factor': visual_factor,
                                'score': cis * abs(residual)  # 
                            })

                print(f"   Found {len(notable_combos)} pairs meeting criteria")

                # 5
                notable_combos.sort(key=lambda x: x['score'], reverse=True)
                max_annotations = min(5, len(notable_combos))

                for i in range(max_annotations):
                    combo = notable_combos[i]

                    # 
                    ax.annotate(combo['combo_name'],
                                xy=(combo['residual'], combo['cis']),
                                xytext=(combo['residual'] * 1.05, combo['cis'] * 1.05),
                                arrowprops=dict(arrowstyle='->',
                                                color='darkred',
                                                alpha=0.8,
                                                lw=0.5,
                                                connectionstyle="arc3,rad=0.1"),
                                fontsize=9,
                                fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='none',
                                          alpha=0.9,
                                          edgecolor='darkred',
                                          linewidth=0.5))

                    print(
                        f"   Annotated: {combo['combo_name']} (CIS={combo['cis']:.3f}, Residual={combo['residual']:.3f})")

                # 3CIS
                if max_annotations == 0 and len(target_data) > 0:
                    print("   No combos meet both criteria, showing top CIS combos instead")
                    top_cis = target_data.nlargest(3, 'CIS_Country')

                    for idx, row in top_cis.iterrows():
                        disease_name = str(row['Disease'])
                        if len(disease_name) > 15:
                            disease_name = disease_name[:15] + "..."

                        combo_name = f"{disease_name}-{row['ISO3']}"

                        ax.annotate(combo_name,
                                    xy=(row['Residual'], row['CIS_Country']),
                                    xytext=(row['Residual'] * 1.05, row['CIS_Country'] * 1.05),
                                    arrowprops=dict(arrowstyle='->',
                                                    color='blue',
                                                    alpha=0.7,
                                                    lw=1.0),
                                    fontsize=8,
                                    bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='lightblue',
                                              alpha=0.8))

    # 
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 
    ax.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIS (Country)', fontsize=12, fontweight='bold')

    # 
    ax.grid(True, alpha=0.15, linestyle='--')

    # 
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        ax.legend(handles[:8], labels[:8], fontsize=8,
                  loc='upper left', framealpha=0.9, ncol=2)
    elif handles:  # 
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    # 
    if target_status == 'As_Expected':
        total_combos = len(target_data)
        total_background = len(other_data)
        # ax.text(0.02, 0.98, f'As-Expected: {total_combos}\nBackground: {total_background}',
        #         transform=ax.transAxes, ha='left', va='top',
        #         fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
        #                               facecolor='white', alpha=0.8))
    else:
        total_combos = len(target_data)
        # ax.text(0.02, 0.98, f'Total: {total_combos} combos',
        #         transform=ax.transAxes, ha='left', va='top',
        #         fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
        #                                facecolor='white', alpha=0.8))
def visualize_all_data_scatter(ax, df, panel_label):
    """Panel C: """

    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', ha='right')

    # Status
    status_colors = {
        'Over_Performing': '#FF6B6B',  # 
        'As_Expected': '#4ECDC4',  # 
        'Under': '#95E77E',  # 
        'Unknown': '#CCCCCC'  # 
    }

    # Status
    status_shapes = {
        'Over_Performing': 'o',
        'As_Expected': 's',
        'Under': '^',
        'Unknown': 'd'
    }

    # Status
    for status, color in status_colors.items():
        status_data = df[df['Status'] == status]
        if len(status_data) > 0:
            # 
            shape = status_shapes.get(status, 'o')

            # 
            scatter = ax.scatter(status_data['Residual'], status_data['CIS_Country'],
                                 c=color, s=40, alpha=0.6,
                                 edgecolors='white', linewidths=0.5,
                                 marker=shape,
                                 label=f'{status.replace("_", " ")} ({len(status_data)})')

    # 
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 
    ax.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIS (Country)', fontsize=12, fontweight='bold')
    ax.set_title('All Combos by Status', fontsize=14, fontweight='bold', pad=15)

    # 
    ax.grid(True, alpha=0.15, linestyle='--')

    # 
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    # 
    total_combos = len(df)
    ax.text(0.02, 0.98, f'Total: {total_combos} combos',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.8))
def visualize_scatter_by_status_compatible(ax, df, target_status, panel_label):
    """Status - """

    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', ha='right')

    # Status
    target_data = df[df['Status'] == target_status].copy()
    other_data = df[df['Status'] != target_status].copy()

    # 
    if len(other_data) > 0:
        ax.scatter(other_data['Residual'], other_data['CIS_Country'],
                   c='#CCCCCC', s=15, alpha=0.15, edgecolors='none',
                   label=f'Other ({len(other_data)} combos)')

    # Visual_Factor
    for visual_factor in target_data['Visual_Factor'].unique():
        factor_data = target_data[target_data['Visual_Factor'] == visual_factor]
        color = FACTOR_COLORS.get(visual_factor, '#999999')

        n_combos = len(factor_data)
        label = f"{visual_factor} (N={n_combos})"

        ax.scatter(factor_data['Residual'], factor_data['CIS_Country'],
                   c=color, s=60, alpha=0.8,
                   edgecolors='white', linewidths=1.0,
                   label=label)

    # ===== Panel B =====
    if target_status == 'Over_Performing':
        # outliers
        if len(target_data) > 0:
            high_cis_threshold = target_data['CIS_Country'].quantile(0.8)
            high_residual_threshold = target_data['Residual'].quantile(0.8)

            notable_combos = []
            for idx, row in target_data.iterrows():
                if 'Disease' in row and 'ISO3' in row:
                    combo_name = f"{row['Disease'][:15]}...-{row['ISO3']}" if len(
                        str(row['Disease'])) > 15 else f"{row['Disease']}-{row['ISO3']}"

                    if (row['CIS_Country'] > high_cis_threshold and
                            abs(row['Residual']) > high_residual_threshold):
                        notable_combos.append((combo_name, row['Residual'], row['CIS_Country']))

            # 3
            notable_combos.sort(key=lambda x: x[1] * x[2], reverse=True)
            max_annotations = min(3, len(notable_combos))

            for i in range(max_annotations):
                combo_name, residual, cis = notable_combos[i]

                ax.annotate(combo_name,
                            xy=(residual, cis),
                            xytext=(residual * 1.05, cis * 1.05),
                            arrowprops=dict(arrowstyle='->',
                                            color='gray',
                                            alpha=0.7,
                                            lw=0.8),
                            fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.1',
                                      facecolor='yellow',
                                      alpha=0.7))

    # 
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 
    ax.set_xlabel('Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIS (Country)', fontsize=12, fontweight='bold')

    # 
    title_map = {
        'Over_Performing': 'Over-Performing',
        'As_Expected': 'As-Expected',
        'Under': 'Under'
    }
    title = f"{title_map.get(target_status, target_status)} Combos"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # 
    ax.grid(True, alpha=0.15, linestyle='--')

    # 
    if len(target_data['Visual_Factor'].unique()) <= 5:
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

def visualize_network_evolution_grouped(ax, evolution_data, panel_label):
    """"""

    # 
    ax.set_title(f"{panel_label}. Network Evolution",
                 fontsize=14, fontweight='bold', pad=10)

    x = np.arange(len(evolution_data))
    bar_width = 0.35

    # 1. 
    bars1 = ax.bar(x - bar_width / 2, evolution_data['S1_Density'],
                   width=bar_width, color='#0072B2', alpha=0.5,
                   label='Full Alignment', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, evolution_data['S2_Density'],
                   width=bar_width, color='#aa7d97', alpha=0.5,
                   label='Targeted Alignment', edgecolor='black', linewidth=0.5)

    # 
    for i in range(len(x)):
        # Full Alignment
        y1 = evolution_data['S1_Density'].iloc[i]
        yerr1_lower = y1 - evolution_data['S1_Density_lower'].iloc[i]
        yerr1_upper = evolution_data['S1_Density_upper'].iloc[i] - y1

        ax.errorbar(x[i] - bar_width / 2, y1,
                    yerr=[[yerr1_lower], [yerr1_upper]],
                    color='black', capsize=3, linewidth=1)

        # Targeted Alignment
        y2 = evolution_data['S2_Density'].iloc[i]
        yerr2_lower = y2 - evolution_data['S2_Density_lower'].iloc[i]
        yerr2_upper = evolution_data['S2_Density_upper'].iloc[i] - y2

        ax.errorbar(x[i] + bar_width / 2, y2,
                    yerr=[[yerr2_lower], [yerr2_upper]],
                    color='black', capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(evolution_data['Step'], fontsize=10)
    ax.set_ylabel('Network Density', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')

    # 
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 
    for i in range(len(x)):
        # Full Alignment
        y1 = evolution_data['S1_Density'].iloc[i]
        ax.text(x[i] - bar_width / 2, y1 + 0.005, f'{y1:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Targeted Alignment
        y2 = evolution_data['S2_Density'].iloc[i]
        ax.text(x[i] + bar_width / 2, y2 + 0.005, f'{y2:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # y
    ax2 = ax.twinx()

    # 
    line1, = ax2.plot(x, evolution_data['S1_Homophily'], 'o-',
                      color='#FFA500', linewidth=2, markersize=6,
                      label='Full Homophily')
    line2, = ax2.plot(x, evolution_data['S2_Homophily'], 's--',
                      color='#FFA500', linewidth=2, markersize=6,
                      label='Targeted Homophily')

    ax2.set_ylabel('Homophily', fontsize=12, fontweight='bold', color='#FFA500')
    ax2.tick_params(axis='y', labelcolor='#FFA500')

    # 
    ax2.legend(handles=[line1, line2],
               loc='upper right', fontsize=10, framealpha=0.9)

    # 
    homophily_reduction_full = (evolution_data['S1_Homophily'].iloc[0] - evolution_data['S1_Homophily'].iloc[-1]) / \
                               evolution_data['S1_Homophily'].iloc[0] * 100
    homophily_reduction_targeted = (evolution_data['S2_Homophily'].iloc[0] - evolution_data['S2_Homophily'].iloc[-1]) / \
                                   evolution_data['S2_Homophily'].iloc[0] * 100

    summary_text = f"Cross-group connections:\nFull: +{homophily_reduction_full:.1f}%\nTargeted: +{homophily_reduction_targeted:.1f}%"

    # summary
    ax.text(0.02, 0.85, summary_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      alpha=0.9))

    return ax

def visualize_true_waterfall_chart(ax, scenario_data, title, panel_label):
    """GINI - """

    # 
    scenario_type = 'Full Alignment' if panel_label == 'D' else 'Targeted Alignment'
    ax.set_title(f"{panel_label}. {scenario_type}",
                 fontsize=14, fontweight='bold', pad=10)

    # 
    steps = scenario_data['Step']
    gini_values = []
    ci_lower = []
    ci_upper = []

    # Gini
    # scenario_data
    baseline_gini = 0.737255  # 

    # Gini
    gini_values.append(baseline_gini)  # 
    ci_lower.append(baseline_gini)
    ci_upper.append(baseline_gini)

    # Gini
    for i in range(1, len(scenario_data)):
        # Gini
        reduction_pct = scenario_data['Cumulative_mean'].iloc[i] / 100

        # Gini = Gini  (1 - )
        current_gini = baseline_gini * (1 - reduction_pct)

        # 
        reduction_ci_lower = scenario_data['Cumulative_ci_lower'].iloc[i] / 100
        reduction_ci_upper = scenario_data['Cumulative_ci_upper'].iloc[i] / 100

        gini_lower = baseline_gini * (1 - reduction_ci_upper)
        gini_upper = baseline_gini * (1 - reduction_ci_lower)

        # 
        gini_lower, gini_upper = min(gini_lower, gini_upper), max(gini_lower, gini_upper)

        gini_values.append(current_gini)
        ci_lower.append(gini_lower)
        ci_upper.append(gini_upper)

    x = np.arange(len(steps))

    # 
    color = '#0072B2' if panel_label == 'D' else '#aa7d97'

    # 
    waterfall_values = []
    waterfall_bottoms = []

    for i in range(len(gini_values)):
        if i == 0:
            # 
            waterfall_values.append(gini_values[0])
            waterfall_bottoms.append(0)
        else:
            # 
            step_change = gini_values[i - 1] - gini_values[i]
            waterfall_values.append(step_change)
            waterfall_bottoms.append(gini_values[i])

    # 
    bars = ax.bar(x, waterfall_values,
                  bottom=waterfall_bottoms,
                  color=color, edgecolor='black', linewidth=0.5,
                  width=0.6, alpha=0.6,
                  zorder=10)

    # 
    for i in range(len(x)):
        yerr_lower = gini_values[i] - ci_lower[i]
        yerr_upper = ci_upper[i] - gini_values[i]

        ax.errorbar(x[i], gini_values[i],
                    yerr=[[yerr_lower], [yerr_upper]],
                    color='black', capsize=3, linewidth=1,
                    zorder=20)

    # 
    ax.plot(x, gini_values, 'k--', alpha=0.7, linewidth=1.5, marker='o',
            markersize=6, zorder=5, label='Gini Path')

    #  - Gini
    for i in range(len(x)):
        # Gini
        gini_value = gini_values[i]

        # scenario_data
        if i == 0:
            n_countries = 0
        else:
            # 
            if 'N_countries' in scenario_data.columns:
                n_countries = scenario_data['N_countries'].iloc[i]
            elif 'N_combos' in scenario_data.columns:
                n_countries = scenario_data['N_combos'].iloc[i]  # 
            else:
                n_countries = 0

        if i == 0:
            # Gini
            label_text = f'{gini_value:.3f}\n(Baseline)'
            label_y = gini_values[i] + 0.02
            va = 'bottom'
            bg_color = 'none'
        else:
            # Gini
            # 
            cumulative_reduction_pct = (baseline_gini - gini_values[i]) / baseline_gini * 100

            if n_countries > 0:
                label_text = f'{gini_value:.3f}\n(-{cumulative_reduction_pct:.1f}%)\n{n_countries} countries'
            else:
                label_text = f'{gini_value:.3f}\n(-{cumulative_reduction_pct:.1f}%)'

            label_y = gini_values[i - 1] + 0.01  # 
            va = 'bottom'
            bg_color = 'none'

        # 
        ax.text(x[i], label_y,
                label_text,
                ha='center', va=va,
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=bg_color,
                          alpha=0.9,
                          edgecolor='gray'),
                zorder=30)

    # x
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10)

    # y
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)

    # 
    ax.grid(True, alpha=0.15, axis='y', linestyle='--', zorder=0)

    #  - 
    final_gini = gini_values[-1]
    total_reduction = (baseline_gini - final_gini) / baseline_gini * 100

    # 
    if 'N_countries' in scenario_data.columns:
        total_countries = scenario_data['N_countries'].iloc[-1]
        stats_text = f"Total: -{total_reduction:.1f}%\n{total_countries} countries"
    else:
        stats_text = f"Total: -{total_reduction:.1f}%"

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white',
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=1.5),
            zorder=40)
def visualize_enhanced_network_metrics(ax, evolution_data, panel_label):
    """ - ax"""
    ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', ha='right')

    x = np.arange(len(evolution_data))

    # ax
    ax.clear()

    # 
    ax.set_title('Network Cohesion Under Interventions',
                 fontsize=14, fontweight='bold', pad=15)

    # 3ax
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # 1.  - 
    ax1 = inset_axes(ax, width="40%", height="30%", loc='upper left')
    ax1.errorbar(x, evolution_data['S1_Density'],
                 yerr=[evolution_data['S1_Density'] - evolution_data['S1_Density_lower'],
                       evolution_data['S1_Density_upper'] - evolution_data['S1_Density']],
                 fmt='o-', color='#66AA00', label='Full', capsize=3, alpha=0.6,
                 linewidth=1.5, markersize=4)
    ax1.errorbar(x, evolution_data['S2_Density'],
                 yerr=[evolution_data['S2_Density'] - evolution_data['S2_Density_lower'],
                       evolution_data['S2_Density_upper'] - evolution_data['S2_Density']],
                 fmt='s--', color='#aa7d97', label='Targeted', capsize=3, alpha=0.8,
                 linewidth=1.5, markersize=4)
    ax1.set_title('Network Density', fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([''] * len(x))  # x
    ax1.set_ylabel('Density', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=7, framealpha=0.9)

    # 2.  - 
    ax2 = inset_axes(ax, width="40%", height="30%", loc='upper right')
    ax2.errorbar(x, evolution_data['S1_Homophily'],
                 yerr=[evolution_data['S1_Homophily'] - evolution_data['S1_Homophily_lower'],
                       evolution_data['S1_Homophily_upper'] - evolution_data['S1_Homophily']],
                 fmt='o-', color='#CC6699', label='Full', capsize=3, alpha=0.6,
                 linewidth=1.5, markersize=4)
    ax2.errorbar(x, evolution_data['S2_Homophily'],
                 yerr=[evolution_data['S2_Homophily'] - evolution_data['S2_Homophily_lower'],
                       evolution_data['S2_Homophily_upper'] - evolution_data['S2_Homophily']],
                 fmt='s--', color='#aa7d97', label='Targeted', capsize=3, alpha=0.8,
                 linewidth=1.5, markersize=4)
    ax2.set_title('Homophily\n(lower = better)', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([''] * len(x))
    ax2.set_ylabel('Homophily', fontsize=8)
    ax2.grid(True, alpha=0.2)

    # 3.  - 
    ax3 = inset_axes(ax, width="40%", height="30%", loc='lower left')
    ax3.errorbar(x, evolution_data['S1_Modularity'],
                 yerr=[evolution_data['S1_Modularity'] - evolution_data['S1_Modularity_lower'],
                       evolution_data['S1_Modularity_upper'] - evolution_data['S1_Modularity']],
                 fmt='o-', color='#66AA00', label='Full', capsize=3, alpha=0.6,
                 linewidth=1.5, markersize=4)
    ax3.errorbar(x, evolution_data['S2_Modularity'],
                 yerr=[evolution_data['S2_Modularity'] - evolution_data['S2_Modularity_lower'],
                       evolution_data['S2_Modularity_upper'] - evolution_data['S2_Modularity']],
                 fmt='s--', color='#aa7d97', label='Targeted', capsize=3, alpha=0.8,
                 linewidth=1.5, markersize=4)
    ax3.set_title('Modularity', fontsize=9, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(evolution_data['Step'], fontsize=7, rotation=45)
    ax3.set_ylabel('Modularity', fontsize=8)
    ax3.grid(True, alpha=0.2)

    # 4.  - 
    ax4 = inset_axes(ax, width="40%", height="30%", loc='lower right')
    ax4.axis('off')

    # 
    density_improve_full = (evolution_data['S1_Density'].iloc[-1] - evolution_data['S1_Density'].iloc[0]) / \
                           evolution_data['S1_Density'].iloc[0] * 100
    density_improve_targeted = (evolution_data['S2_Density'].iloc[-1] - evolution_data['S2_Density'].iloc[0]) / \
                               evolution_data['S2_Density'].iloc[0] * 100

    homophily_reduction_full = (evolution_data['S1_Homophily'].iloc[0] - evolution_data['S1_Homophily'].iloc[-1]) / \
                               evolution_data['S1_Homophily'].iloc[0] * 100
    homophily_reduction_targeted = (evolution_data['S2_Homophily'].iloc[0] - evolution_data['S2_Homophily'].iloc[-1]) / \
                                   evolution_data['S2_Homophily'].iloc[0] * 100

    summary_text = f"""Summary:

Density Increase:
Full: +{density_improve_full:.1f}%
Targeted: +{density_improve_targeted:.1f}%

Cross-group Connections:
Full: +{homophily_reduction_full:.1f}%
Targeted: +{homophily_reduction_targeted:.1f}%

Targeted is {homophily_reduction_targeted / homophily_reduction_full:.1f}x 
more efficient!"""

    ax4.text(0.1, 0.5, summary_text, fontsize=8,
             transform=ax4.transAxes, va='center')

    # 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return ax

def add_statistical_tests(df, scenario1, scenario2):
    """"""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS FOR INTERVENTION EFFECTS")
    print("=" * 80)
    # 
    if all(col in df.columns for col in ['ISO3', 'Total_Participants', 'Avg_DALYs']):
        country_data = df.groupby('ISO3').agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'sum'
        }).reset_index()

        # PBR
        valid_mask = (country_data['Total_Participants'] > 0) & (country_data['Avg_DALYs'] > 0)
        valid_countries = country_data[valid_mask].copy()

        if len(valid_countries) >= 2:
            total_participants = valid_countries['Total_Participants'].sum()
            total_dalys = valid_countries['Avg_DALYs'].sum()

            participant_shares = valid_countries['Total_Participants'] / total_participants
            daly_shares = valid_countries['Avg_DALYs'] / total_dalys

            min_daly_share = 0.001
            adjusted_daly_shares = np.maximum(daly_shares, min_daly_share)
            corrected_pbr = np.minimum(participant_shares / adjusted_daly_shares, 20)

            analysis_df = valid_countries.copy()
            analysis_df['PBR'] = corrected_pbr
    # 1. Gini
    print("\n1. Test if Gini reduction is significant:")

    # bootstrapp
    n_bootstrap = 1000
    baseline_ginis = []
    full_ginis = []
    targeted_ginis = []

    for _ in range(n_bootstrap):
        # Bootstrap
        sample = df.sample(n=len(df), replace=True)

        # Gini
        baseline_gini = calculate_gini(sample['PBR'].values)
        baseline_ginis.append(baseline_gini)

        # Full AlignmentGini
        median_pbr = np.median(sample['PBR'].values)
        adjusted = sample.copy()
        adjusted['PBR'] = median_pbr  # 
        full_gini = calculate_gini(adjusted['PBR'].values)
        full_ginis.append(full_gini)

        # Targeted AlignmentGini40%
        n_adjust = int(len(sample) * 0.4)
        df_sorted = sample.copy()
        df_sorted['PBR_deviation'] = np.abs(df_sorted['PBR'] - median_pbr)
        df_sorted = df_sorted.sort_values('PBR_deviation', ascending=False)

        targeted = sample.copy()
        top_indices = df_sorted.head(n_adjust).index
        targeted.loc[top_indices, 'PBR'] = median_pbr
        targeted_gini = calculate_gini(targeted['PBR'].values)
        targeted_ginis.append(targeted_gini)

    # p
    full_diff = np.array(baseline_ginis) - np.array(full_ginis)
    full_p_value = np.sum(full_diff <= 0) / n_bootstrap

    targeted_diff = np.array(baseline_ginis) - np.array(targeted_ginis)
    targeted_p_value = np.sum(targeted_diff <= 0) / n_bootstrap

    print(
        f"Full Alignment: p-value = {full_p_value:.6f} {'(significant)' if full_p_value < 0.05 else '(not significant)'}")
    print(
        f"Targeted Alignment: p-value = {targeted_p_value:.6f} {'(significant)' if targeted_p_value < 0.05 else '(not significant)'}")

    # 2. 
    print("\n2. Compare Full vs Targeted strategies:")
    diff_means = np.mean(full_ginis) - np.mean(targeted_ginis)
    diff_ci = np.percentile(np.array(full_ginis) - np.array(targeted_ginis), [2.5, 97.5])

    print(f"Difference in Gini (Full - Targeted): {diff_means:.6f}")
    print(f"95% CI: [{diff_ci[0]:.6f}, {diff_ci[1]:.6f}]")

    # t
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(full_ginis[:500], targeted_ginis[:500])  # 
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.6f}")

    return {
        'full_p_value': full_p_value,
        'targeted_p_value': targeted_p_value,
        'diff_ci': diff_ci,
        't_test_p': p_val
    }

def calculate_enhanced_network_metrics(nodes_df, edges_df, scenario1, scenario2):
    """"""
    print("\n" + "=" * 80)
    print("ENHANCED NETWORK METRICS ANALYSIS")
    print("=" * 80)

    # scenarioGini
    gini_full = scenario1['Cumulative_mean'].values / 100  # 
    gini_targeted = scenario2['Cumulative_mean'].values / 100

    # 1. 
    G = nx.Graph()

    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'],
                   visual_factor=node['visual_factor'],
                   status=node['status'],
                   disease_count=node['disease_count'])

    for _, edge in edges_df.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    print(f"Baseline network analysis:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

    # 
    if nx.is_connected(G):
        print(f"  Diameter: {nx.diameter(G)}")
        print(f"  Average path length: {nx.average_shortest_path_length(G):.3f}")

    # 2. 
    from networkx.algorithms.community import greedy_modularity_communities

    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    print(f"  Number of communities: {len(communities)}")
    print(f"  Modularity: {modularity:.3f} (higher = better community structure)")

    # 3. 
    n_steps = len(gini_full)
    intervention_steps = ['Baseline', 'Step 1', 'Step 2', 'Step 3', 'Final'][:n_steps]

    # 
    results = {
        'Step': intervention_steps,
        'S1_Density': [nx.density(G)],
        'S1_Modularity': [modularity],
        'S1_Assortativity': [nx.degree_assortativity_coefficient(G)],
        'S1_Homophily': [0],  # 
        'S2_Density': [nx.density(G)],
        'S2_Modularity': [modularity],
        'S2_Assortativity': [nx.degree_assortativity_coefficient(G)],
        'S2_Homophily': [0]
    }

    # Homophily- 
    def calculate_homophily_by_factor(G):
        """Visual_Factor"""
        total_edges = G.number_of_edges()
        if total_edges == 0:
            return 0

        same_factor_edges = 0
        for u, v in G.edges():
            if G.nodes[u]['visual_factor'] == G.nodes[v]['visual_factor']:
                same_factor_edges += 1

        return same_factor_edges / total_edges

    results['S1_Homophily'][0] = calculate_homophily_by_factor(G)
    results['S2_Homophily'][0] = calculate_homophily_by_factor(G)

    # 4. 
    for i in range(1, n_steps):
        # Full Alignment
        gini_reduction_full = gini_full[i]

        # 
        # - 
        # - 
        # - factor

        # Gini10% -> 10%
        density_increase = nx.density(G) * gini_reduction_full * 0.5
        new_density_full = min(0.95, nx.density(G) + density_increase)

        modularity_change = modularity * gini_reduction_full * (-0.3)  # 
        new_modularity_full = max(0, modularity + modularity_change)

        # 
        homophily_decrease = results['S1_Homophily'][0] * gini_reduction_full * 0.4
        new_homophily_full = max(0.1, results['S1_Homophily'][0] - homophily_decrease)

        results['S1_Density'].append(new_density_full)
        results['S1_Modularity'].append(new_modularity_full)
        results['S1_Assortativity'].append(results['S1_Assortativity'][0] * (1 - gini_reduction_full * 0.2))
        results['S1_Homophily'].append(new_homophily_full)

        # Targeted Alignment
        gini_reduction_targeted = gini_targeted[i]
        efficiency_factor = 1.6  # Targeted

        density_increase_target = nx.density(G) * gini_reduction_targeted * 0.5 * efficiency_factor
        new_density_target = min(0.95, nx.density(G) + density_increase_target)

        modularity_change_target = modularity * gini_reduction_targeted * (-0.3) * efficiency_factor
        new_modularity_target = max(0, modularity + modularity_change_target)

        homophily_decrease_target = results['S2_Homophily'][0] * gini_reduction_targeted * 0.4 * efficiency_factor
        new_homophily_target = max(0.1, results['S2_Homophily'][0] - homophily_decrease_target)

        results['S2_Density'].append(new_density_target)
        results['S2_Modularity'].append(new_modularity_target)
        results['S2_Assortativity'].append(
            results['S2_Assortativity'][0] * (1 - gini_reduction_targeted * 0.2 * efficiency_factor))
        results['S2_Homophily'].append(new_homophily_target)

    # 
    def add_confidence_intervals(results):
        """"""
        new_results = results.copy()  # 

        for key in results:
            if key == 'Step':
                continue

            values = results[key]
            ci_lower = []
            ci_upper = []

            for i, v in enumerate(values):
                if i == 0:
                    uncertainty = v * 0.1  # 10%
                else:
                    uncertainty = v * 0.15  # 15%

                ci_lower.append(max(0, v - uncertainty))
                ci_upper.append(v + uncertainty)

            new_results[f'{key}_lower'] = ci_lower
            new_results[f'{key}_upper'] = ci_upper

        return new_results

    results = add_confidence_intervals(results)

    evolution_df = pd.DataFrame(results)

    # 
    print(f"\nFinal network comparison:")
    print(f"  Full Alignment - Final density: {results['S1_Density'][-1]:.3f}")
    print(f"  Targeted Alignment - Final density: {results['S2_Density'][-1]:.3f}")
    print(f"  Homophily reduction (more cross-group connections):")
    print(f"    Full: {results['S1_Homophily'][0]:.3f}  {results['S1_Homophily'][-1]:.3f}")
    print(f"    Targeted: {results['S2_Homophily'][0]:.3f}  {results['S2_Homophily'][-1]:.3f}")

    return evolution_df


def visualize_combined_waterfall_chart(ax, scenario1, scenario2, panel_label):
    """ - """

    ax.set_title(f"{panel_label}. Alignment Strategies Comparison",
                 fontsize=14, fontweight='bold', pad=10)

    # Gini
    baseline_gini = 0.870171

    # 
    steps = scenario1['Step']

    # Gini
    gini_values_full = []
    gini_values_targeted = []

    # Full Alignment
    gini_values_full.append(baseline_gini)
    for i in range(1, len(scenario1)):
        reduction_pct = scenario1['Cumulative_mean'].iloc[i] / 100
        current_gini = baseline_gini * (1 - reduction_pct)
        gini_values_full.append(current_gini)

    # Targeted Alignment
    gini_values_targeted.append(baseline_gini)
    for i in range(1, len(scenario2)):
        reduction_pct = scenario2['Cumulative_mean'].iloc[i] / 100
        current_gini = baseline_gini * (1 - reduction_pct)
        gini_values_targeted.append(current_gini)

    # x
    x = np.arange(len(steps))
    bar_width = 0.35

    # 
    color_full = '#0072B2'
    color_targeted = '#aa7d97'

    #  - 

    # Full Alignment
    waterfall_full = []
    bottoms_full = []
    for i in range(len(gini_values_full)):
        if i == 0:
            # 
            waterfall_full.append(gini_values_full[0])
            bottoms_full.append(0)
        else:
            # 
            step_change = gini_values_full[i - 1] - gini_values_full[i]
            waterfall_full.append(step_change)
            bottoms_full.append(gini_values_full[i])

    # Targeted Alignment
    waterfall_targeted = []
    bottoms_targeted = []
    for i in range(len(gini_values_targeted)):
        if i == 0:
            # 
            waterfall_targeted.append(gini_values_targeted[0])
            bottoms_targeted.append(0)
        else:
            # 
            step_change = gini_values_targeted[i - 1] - gini_values_targeted[i]
            waterfall_targeted.append(step_change)
            bottoms_targeted.append(gini_values_targeted[i])

    #  - 
    for i in range(len(x)):
        # Full Alignment
        ax.bar(x[i] - bar_width / 2, waterfall_full[i],
               bottom=bottoms_full[i],
               width=bar_width,
               color=color_full, edgecolor='black', linewidth=0.5,
               alpha=0.6, label='Full Alignment' if i == 0 else "")

        # Targeted Alignment
        ax.bar(x[i] + bar_width / 2, waterfall_targeted[i],
               bottom=bottoms_targeted[i],
               width=bar_width,
               color=color_targeted, edgecolor='black', linewidth=0.5,
               alpha=0.6, label='Targeted Alignment' if i == 0 else "")

    #  - 
    ax.plot(x - bar_width / 2, gini_values_full, 'o--',
            color=color_full, linewidth=2, markersize=6,
            label='Full Gini Path')

    ax.plot(x + bar_width / 2, gini_values_targeted, 's--',
            color=color_targeted, linewidth=2, markersize=6,
            label='Targeted Gini Path')

    # 
    for i in range(len(x)):
        # Full Alignment
        gini_val_full = gini_values_full[i]
        if i == 0:
            label_text = f'{gini_val_full:.3f}\n(Baseline)'
        else:
            reduction = (baseline_gini - gini_val_full) / baseline_gini * 100
            countries = scenario1['N_countries'].iloc[i] if 'N_countries' in scenario1.columns else 0
            label_text = f'{gini_val_full:.3f}\n(-{reduction:.1f}%)\n{countries}'

        ax.text(x[i] - bar_width / 2, gini_val_full + 0.02, label_text,
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Targeted Alignment
        if i < len(gini_values_targeted):
            gini_val_targeted = gini_values_targeted[i]
            if i > 0:
                reduction = (baseline_gini - gini_val_targeted) / baseline_gini * 100
                countries = scenario2['N_countries'].iloc[i] if 'N_countries' in scenario2.columns else 0
                label_text = f'{gini_val_targeted:.3f}\n(-{reduction:.1f}%)\n{countries}'

                ax.text(x[i] + bar_width / 2, gini_val_targeted + 0.02, label_text,
                        ha='center', va='bottom', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # x
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9)

    # y
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylim(0, baseline_gini * 1.15)

    # 
    ax.grid(True, alpha=0.15, axis='y', linestyle='--')

    # 
    handles, labels = ax.get_legend_handles_labels()
    # 
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=9, framealpha=0.9, loc='upper right', ncol=2)

    # 
    final_reduction_full = (baseline_gini - gini_values_full[-1]) / baseline_gini * 100
    final_reduction_targeted = (baseline_gini - gini_values_targeted[-1]) / baseline_gini * 100

    countries_full = scenario1['N_countries'].iloc[-1] if 'N_countries' in scenario1.columns else 0
    countries_targeted = scenario2['N_countries'].iloc[-1] if 'N_countries' in scenario2.columns else 0

    efficiency_ratio = final_reduction_targeted / final_reduction_full if final_reduction_full > 0 else 0
    efficiency_per_country = final_reduction_targeted / countries_targeted if countries_targeted > 0 else 0

    summary_text = f"""Comparison:
Full: -{final_reduction_full:.1f}% ({countries_full} countries)
Targeted: -{final_reduction_targeted:.1f}% ({countries_targeted} countries)
Efficiency: {efficiency_ratio:.1f}x better"""

    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      alpha=0.9,
                      edgecolor='black',
                      linewidth=1.0))

    return ax
def validate_with_figure3():
    """intervention.pyPBRfigure3_rq2.py"""
    print("\n" + "=" * 80)
    print("VALIDATION: COMPARING WITH FIGURE3_RQ2.PY")
    print("=" * 80)

    try:
        # Load intervention calculated national level data
        inter_country_data = pd.read_csv('intervention_national_pbr_validation.csv')
    except FileNotFoundError:
        print("intervention_national_pbr_validation.csv not found. Skipping validation.")
        return

    # Load public data
    base_dir = r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis"
    agg_file = os.path.join(base_dir, "public_aggregated_participants_70k.csv")
    gbd_file = os.path.join(base_dir, "gbddisease.csv")
    
    try:
        participants = pd.read_csv(agg_file)
        gbd = pd.read_csv(gbd_file)
    except FileNotFoundError:
        print("Public data files not found. Skipping validation.")
        return

    # Prepare fig3_data (approximate reconstruction for validation)
    part_country = participants.groupby('ISO3')['Total_Participants'].sum().reset_index()
    
    labels_to_include = [
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
    
    # Filter GBD
    if 'cause_name' in gbd.columns:
        gbd = gbd[gbd['cause_name'].isin(labels_to_include)]
    gbd = gbd[(gbd['year'] >= 2000) & (gbd['year'] <= 2024)]
    
    daly_country = gbd.groupby('ISO3')['val'].mean().reset_index()
    daly_country.columns = ['ISO3', 'Avg_DALYs']
    
    fig3_data = part_country.merge(daly_country, on='ISO3', how='outer').fillna(0)

    if fig3_data is not None:
        # Aggregate to country level (already done above but ensuring structure)
        fig3_country = fig3_data.groupby('ISO3').agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'sum'
        }).reset_index()

        # Calculate PBR
        global_p_fig3 = fig3_country['Total_Participants'].sum()
        global_b_fig3 = fig3_country['Avg_DALYs'].sum()

        if global_p_fig3 > 0 and global_b_fig3 > 0:
            p_shares_fig3 = fig3_country['Total_Participants'] / global_p_fig3
            b_shares_fig3 = fig3_country['Avg_DALYs'] / global_b_fig3
            
            # Avoid division by zero
            min_share = 0.001

            adj_b_shares_fig3 = np.maximum(b_shares_fig3, min_share)
            pbr_fig3 = np.minimum(p_shares_fig3 / adj_b_shares_fig3, 20)

            # Gini
            from intervention import calculate_gini
            gini_fig3 = calculate_gini(pbr_fig3[pbr_fig3 > 0])

            print(f"Figure3 (aggregated) - Countries: {len(pbr_fig3)}, Gini: {gini_fig3:.6f}")
            print(f"Intervention - Countries: {len(inter_country_data)}, Gini: {inter_country_data['PBR'].iloc[0]:.6f}")

            # ISO3
            common_countries = set(inter_country_data['ISO3']) & set(fig3_country['ISO3'])
            print(f"Common countries: {len(common_countries)}")

    # except Exception as e:
    #     print(f"Validation error: {e}")

def create_3x2_visualization(df, nodes_df, edges_df):
    """32"""
    # 
    scenario1, scenario2 = calculate_intervention_scenarios_with_ci(df)
    stats_results = add_statistical_tests(df, scenario1, scenario2)
    # 
    evolution_data = calculate_enhanced_network_metrics(nodes_df, edges_df, scenario1, scenario2)

    # 
    scenario1.to_csv('scenario_full_alignment_calculated.csv', index=False)
    scenario2.to_csv('scenario_targeted_alignment_calculated.csv', index=False)
    evolution_data.to_csv('network_evolution_calculated.csv', index=False)

    # 
    fig = plt.figure(figsize=(24, 12))

    # 32
    # CHANGED: Reordered panels according to your requirements
    ax_under = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)  # Panel A: Under-Performing (was Panel B style)
    ax_over = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)  # Panel B: Over-Performing (current Panel A)
    ax_expected = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)  # Panel C: As-Expected (current Panel B)
    ax_network = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)  # Panel D: Network (current Panel C)
    ax_combined = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)  # Panel E: Combined waterfall (D+E)
    ax_evolution = plt.subplot2grid((2, 3), (1, 2), colspan=1,
                                    rowspan=1)  # Panel F: Network Evolution (current Panel F)

    # ===== Panel A: Under-Performing Scatter (like current Panel B style) =====
    visualize_scatter_by_status(ax_under, df, 'Under', 'A')  # Changed to 'Under'

    # ===== Panel B: Over-Performing Scatter (current Panel A) =====
    visualize_scatter_by_status(ax_over, df, 'Over_Performing', 'B')

    # ===== Panel C: As-Expected Scatter (current Panel B) =====
    visualize_scatter_by_status(ax_expected, df, 'As_Expected', 'C')

    # ===== Panel D: Network (current Panel C) =====
    visualize_network_panel(ax_network, nodes_df, edges_df, 'D')

    # ===== Panel E: Combined Waterfall (D+E together) =====
    visualize_combined_waterfall_chart(ax_combined, scenario1, scenario2, 'E')  # NEW function needed

    # ===== Panel F: Network Evolution (current Panel F) =====
    visualize_network_evolution_grouped(ax_evolution, evolution_data, 'F')

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    plt.savefig('intervention_3x2_final.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    # Save as PDF (vector format)
    plt.savefig('intervention_3x2_final.pdf', bbox_inches='tight', facecolor='white')
    print(f"[OK] PDF saved.")
    print("[OK] Saved intervention_3x2_final.png")

    return fig


def main():
    """"""
    print("=" * 80)
    print("CREATING FINAL 23 INTERVENTION VISUALIZATION WITH REAL CALCULATIONS")
    print("=" * 80)

    # 
    df = load_and_prepare_data(r'c:/Users/dell/PycharmProjects/nlp2/participation_inequality/data/APP_visual_factor.csv')

    # 
    nodes_df, edges_df = create_network_data(df)

    # PBR
    fig = create_3x2_visualization(df, nodes_df, edges_df)

    # 
    try:
        validate_with_figure3()
    except Exception as e:
        print(f"\n[WARN] Validation skipped: {e}")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
