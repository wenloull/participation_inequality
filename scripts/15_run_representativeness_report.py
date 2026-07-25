import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set style and color palettes
plt.style.use('default')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Dataset name mapping for the 5-tier structure
DATASET_NAMES = {
    '301k': 'FullRCT',
    '195k': 'TotalRCT',
    '137k': 'DisTSub',
    '124k': 'GeoTSub',
    '100k': 'DisGeoSub'
}

# Dataset colors using mapped names
dataset_colors = {
    'FullRCT': COLORS['primary_blue'],
    'TotalRCT': COLORS['primary_red'],
    'DisTSub': COLORS['accent_orange'],
    'GeoTSub': COLORS['dark_green'],
    'DisGeoSub': COLORS['warning_orange']
}

def cramers_v(confusion_matrix):
    """Calculate Cramér's V effect size from confusion matrix with bias correction"""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def interpret_cramers_v(v):
    """Interpret Cramér's V effect size"""
    if v < 0.1:
        return "Very Small (Highly Representative)"
    elif v < 0.3:
        return "Small (Representative)"
    elif v < 0.5:
        return "Medium (Moderately Representative)"
    else:
        return "Large (Not Representative)"

def calculate_percentage_differences(baseline_dist, subset_dist):
    """Calculate percentage differences between distributions"""
    all_categories = sorted(set(baseline_dist.index) | set(subset_dist.index))
    baseline_aligned = baseline_dist.reindex(all_categories, fill_value=0)
    subset_aligned = subset_dist.reindex(all_categories, fill_value=0)

    b_sum = baseline_aligned.sum()
    s_sum = subset_aligned.sum()

    baseline_pct = (baseline_aligned / b_sum * 100) if b_sum > 0 else baseline_aligned
    subset_pct = (subset_aligned / s_sum * 100) if s_sum > 0 else subset_aligned

    abs_diff = np.abs(baseline_pct - subset_pct)

    return {
        'max_diff': abs_diff.max(),
        'mean_diff': abs_diff.mean(),
        'median_diff': abs_diff.median(),
        'baseline_pct': baseline_pct,
        'subset_pct': subset_pct,
        'abs_diff': abs_diff
    }

def get_mesh_high_level_category(tree_number):
    """Extract high-level MeSH category from tree number"""
    if pd.isna(tree_number) or str(tree_number).strip() == '':
        return 'Unknown'
    category_map = {
        'A': 'Anatomy',
        'B': 'Organisms',
        'C': 'Diseases',
        'D': 'Chemicals and Drugs',
        'E': 'Analytical, Diagnostic and Therapeutic Techniques',
        'F': 'Psychiatry and Psychology',
        'G': 'Phenomena and Processes',
        'H': 'Disciplines and Occupations',
        'I': 'Anthropology, Education, Sociology and Social Phenomena',
        'J': 'Technology, Industry, Agriculture',
        'K': 'Humanities',
        'L': 'Information Science',
        'M': 'Named Groups',
        'N': 'Health Care',
        'V': 'Publication Characteristics',
        'Z': 'Geographicals'
    }
    first_char = str(tree_number)[0].upper()
    return category_map.get(first_char, 'Unknown')

print("🚀 Starting Updated 5-Tier Representativeness Analysis...")
print("=" * 80)

# =============================================================================
# SECTION 1: LOAD DATASETS FOR THE 5 STRUCTURE LEVELS
# =============================================================================

print("\n🔄 Loading base reference data from data/ and public/...")

# Base raw files in data/
year_301k = pd.read_csv('data/year_301k.csv')
year_195k = pd.read_csv('data/year_195k.csv')
author_301k = pd.read_csv('data/author_301k.csv')
author_195k = pd.read_csv('data/author_195k.csv')
journal_301k = pd.read_csv('data/journal_301k.csv')
journal_195k = pd.read_csv('data/journal_195k.csv')
mesh_301k = pd.read_csv('data/mesh_301k.csv')
mesh_195k = pd.read_csv('data/mesh_195k.csv')

# Load PMIDs defining the 5 levels
pmid_301k = set(year_301k['PMID'])
pmid_195k = set(year_195k['PMID'])
pmid_137k = set(pd.read_csv('data/year_138k.csv')['PMID'])

# New 124k dataset from public/geoinfor183_disease_matched.csv
geoinfor183 = pd.read_csv('public/geoinfor183_disease_matched.csv')
pmid_124k = set(geoinfor183['PMID'])

# 100k dataset (legacy DisGeoSub / complete subset)
pmid_100k = set(pd.read_csv('data/year_99k.csv')['PMID'])

print(f"  • 301k (FullRCT):   {len(pmid_301k):,d} PMIDs")
print(f"  • 195k (TotalRCT):  {len(pmid_195k):,d} PMIDs")
print(f"  • 137k (DisTSub):   {len(pmid_137k):,d} PMIDs")
print(f"  • 124k (GeoTSub):   {len(pmid_124k):,d} PMIDs")
print(f"  • 100k (DisGeoSub): {len(pmid_100k):,d} PMIDs")

# Construct files dictionary for each dataset level
datasets_files = {
    '301k': {
        'year': year_301k,
        'author': author_301k,
        'journal': journal_301k,
        'mesh': mesh_301k
    },
    '195k': {
        'year': year_195k,
        'author': author_195k,
        'journal': journal_195k,
        'mesh': mesh_195k
    },
    '137k': {
        'year': year_195k[year_195k['PMID'].isin(pmid_137k)],
        'author': author_195k[author_195k['PMID'].isin(pmid_137k)],
        'journal': journal_195k[journal_195k['PMID'].isin(pmid_137k)],
        'mesh': mesh_195k[mesh_195k['PMID'].isin(pmid_137k)]
    },
    '124k': {
        'year': year_195k[year_195k['PMID'].isin(pmid_124k)],
        'author': author_195k[author_195k['PMID'].isin(pmid_124k)],
        'journal': journal_195k[journal_195k['PMID'].isin(pmid_124k)],
        'mesh': mesh_195k[mesh_195k['PMID'].isin(pmid_124k)]
    },
    '100k': {
        'year': year_195k[year_195k['PMID'].isin(pmid_100k)],
        'author': author_195k[author_195k['PMID'].isin(pmid_100k)],
        'journal': journal_195k[journal_195k['PMID'].isin(pmid_100k)],
        'mesh': mesh_195k[mesh_195k['PMID'].isin(pmid_100k)]
    }
}

# =============================================================================
# SECTION 2: DATASET OVERVIEW SUMMARY
# =============================================================================

print("\n📊 Creating Dataset Overview Summary...")
overview_stats = []

for code, files in datasets_files.items():
    name = DATASET_NAMES[code]
    y_df = files['year']
    pmid_col = 'PMID' if 'PMID' in y_df.columns else 'pmid'
    year_col = 'YEAR' if 'YEAR' in y_df.columns else 'PY'

    unique_studies = y_df[pmid_col].nunique()
    year_range = f"{y_df[year_col].min()}-{y_df[year_col].max()}"
    unique_years = y_df[year_col].nunique()

    unique_countries = files['author']['ISO3'].nunique()
    unique_journals = files['journal']['TA'].nunique()
    unique_categories = files['journal']['Category'].nunique()
    mesh_col = 'treenumber' if 'treenumber' in files['mesh'].columns else 'Mesh'
    unique_mesh = files['mesh'][mesh_col].dropna().nunique()

    overview_stats.append({
        'Dataset_Code': code,
        'Dataset': name,
        'Unique_Studies': unique_studies,
        'Year_Range': year_range,
        'Unique_Years': unique_years,
        'Unique_Countries': unique_countries,
        'Unique_Journals': unique_journals,
        'Journal_Categories': unique_categories,
        'Unique_MeSH': unique_mesh
    })

overview_df = pd.DataFrame(overview_stats)
print("\n📈 Complete Dataset Overview Summary:")
print(overview_df.to_string(index=False))

# Save overview stats
overview_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_overview_summary.csv'), index=False)

# =============================================================================
# SECTION 3: ANALYTICAL DOMAINS COMPARISONS
# =============================================================================

# Define the key level comparisons
COMPARISONS = [
    ('301k', '195k', 'FullRCT vs TotalRCT'),
    ('195k', '137k', 'TotalRCT vs DisTSub'),
    ('195k', '124k', 'TotalRCT vs GeoTSub'),
    ('195k', '100k', 'TotalRCT vs DisGeoSub')
]

# --- 3.1 Year Analysis ---
def analyze_year(base_code, sub_code, label):
    b_df = datasets_files[base_code]['year']
    s_df = datasets_files[sub_code]['year']
    b_col = 'YEAR' if 'YEAR' in b_df.columns else 'PY'
    s_col = 'YEAR' if 'YEAR' in s_df.columns else 'PY'
    b_pmid = 'PMID' if 'PMID' in b_df.columns else 'pmid'
    s_pmid = 'PMID' if 'PMID' in s_df.columns else 'pmid'

    b_years = b_df[b_col].nunique()
    s_years = s_df[s_col].nunique()
    b_studies = b_df[b_pmid].nunique()
    s_studies = s_df[s_pmid].nunique()

    year_cov = s_years / b_years * 100 if b_years > 0 else 0
    study_cov = s_studies / b_studies * 100 if b_studies > 0 else 0

    b_dist = b_df[b_col].value_counts().sort_index()
    s_dist = s_df[s_col].value_counts().sort_index()
    common_years = sorted(set(b_dist.index) & set(s_dist.index))

    b_comm = b_dist.reindex(common_years, fill_value=0)
    s_comm = s_dist.reindex(common_years, fill_value=0)

    tbl = np.array([b_comm.values, s_comm.values])
    mask = (tbl[0] > 0) | (tbl[1] > 0)
    tbl = tbl[:, mask]

    v = cramers_v(tbl) if tbl.shape[1] > 1 else 0.0
    interp = interpret_cramers_v(v)
    pct = calculate_percentage_differences(b_comm, s_comm)

    return {
        'comparison': label,
        'base_count': b_years,
        'sub_count': s_years,
        'count_cov': year_cov,
        'base_total': b_studies,
        'sub_total': s_studies,
        'total_cov': study_cov,
        'cramers_v': v,
        'interp': interp,
        'pct_diff': pct,
        'b_dist': b_dist,
        's_dist': s_dist
    }

# --- 3.2 Author Analysis ---
def analyze_author(base_code, sub_code, label):
    b_df = datasets_files[base_code]['author']
    s_df = datasets_files[sub_code]['author']

    b_cnt = b_df['ISO3'].nunique()
    s_cnt = s_df['ISO3'].nunique()
    b_tot = len(b_df)
    s_tot = len(s_df)

    cnt_cov = s_cnt / b_cnt * 100 if b_cnt > 0 else 0
    tot_cov = s_tot / b_tot * 100 if b_tot > 0 else 0

    b_dist = b_df['ISO3'].value_counts()
    s_dist = s_df['ISO3'].value_counts()
    all_cnt = sorted(set(b_dist.index) | set(s_dist.index))

    b_alg = b_dist.reindex(all_cnt, fill_value=0)
    s_alg = s_dist.reindex(all_cnt, fill_value=0)

    tbl = np.array([b_alg.values, s_alg.values])
    mask = (tbl[0] > 0) | (tbl[1] > 0)
    tbl = tbl[:, mask]

    v = cramers_v(tbl) if tbl.shape[1] > 1 else 0.0
    interp = interpret_cramers_v(v)
    pct = calculate_percentage_differences(b_dist, s_dist)

    return {
        'comparison': label,
        'base_count': b_cnt,
        'sub_count': s_cnt,
        'count_cov': cnt_cov,
        'base_total': b_tot,
        'sub_total': s_tot,
        'total_cov': tot_cov,
        'cramers_v': v,
        'interp': interp,
        'pct_diff': pct,
        'b_dist': b_dist,
        's_dist': s_dist
    }

# --- 3.3 Journal Analysis ---
def analyze_journal(base_code, sub_code, label):
    b_df = datasets_files[base_code]['journal']
    s_df = datasets_files[sub_code]['journal']

    b_j = b_df['TA'].nunique()
    s_j = s_df['TA'].nunique()
    b_c = b_df['Category'].nunique()
    s_c = s_df['Category'].nunique()
    b_tot = len(b_df)
    s_tot = len(s_df)

    j_cov = s_j / b_j * 100 if b_j > 0 else 0
    tot_cov = s_tot / b_tot * 100 if b_tot > 0 else 0

    b_dist = b_df['Category'].value_counts()
    s_dist = s_df['Category'].value_counts()
    all_cats = sorted(set(b_dist.index) | set(s_dist.index))

    b_alg = b_dist.reindex(all_cats, fill_value=0)
    s_alg = s_dist.reindex(all_cats, fill_value=0)

    tbl = np.array([b_alg.values, s_alg.values])
    mask = (tbl[0] > 0) | (tbl[1] > 0)
    tbl = tbl[:, mask]

    v = cramers_v(tbl) if tbl.shape[1] > 1 else 0.0
    interp = interpret_cramers_v(v)
    pct = calculate_percentage_differences(b_dist, s_dist)

    return {
        'comparison': label,
        'base_count': b_j,
        'sub_count': s_j,
        'count_cov': j_cov,
        'base_total': b_tot,
        'sub_total': s_tot,
        'total_cov': tot_cov,
        'cramers_v': v,
        'interp': interp,
        'pct_diff': pct,
        'b_dist': b_dist,
        's_dist': s_dist
    }

# --- 3.4 MeSH Analysis ---
def analyze_mesh(base_code, sub_code, label):
    b_df = datasets_files[base_code]['mesh'].copy()
    s_df = datasets_files[sub_code]['mesh'].copy()
    b_col = 'treenumber' if 'treenumber' in b_df.columns else 'Mesh'
    s_col = 'treenumber' if 'treenumber' in s_df.columns else 'Mesh'

    b_df['HighLevelCategory'] = b_df[b_col].apply(get_mesh_high_level_category)
    s_df['HighLevelCategory'] = s_df[s_col].apply(get_mesh_high_level_category)

    b_m = b_df[b_col].dropna().nunique()
    s_m = s_df[s_col].dropna().nunique()
    b_tot = len(b_df)
    s_tot = len(s_df)

    m_cov = s_m / b_m * 100 if b_m > 0 else 0
    tot_cov = s_tot / b_tot * 100 if b_tot > 0 else 0

    b_dist = b_df['HighLevelCategory'].value_counts()
    s_dist = s_df['HighLevelCategory'].value_counts()
    all_cats = sorted(set(b_dist.index) | set(s_dist.index))

    b_alg = b_dist.reindex(all_cats, fill_value=0)
    s_alg = s_dist.reindex(all_cats, fill_value=0)

    tbl = np.array([b_alg.values, s_alg.values])
    mask = (tbl[0] > 0) | (tbl[1] > 0)
    tbl = tbl[:, mask]

    v = cramers_v(tbl) if tbl.shape[1] > 1 else 0.0
    interp = interpret_cramers_v(v)
    pct = calculate_percentage_differences(b_dist, s_dist)

    return {
        'comparison': label,
        'base_count': b_m,
        'sub_count': s_m,
        'count_cov': m_cov,
        'base_total': b_tot,
        'sub_total': s_tot,
        'total_cov': tot_cov,
        'cramers_v': v,
        'interp': interp,
        'pct_diff': pct,
        'b_dist': b_dist,
        's_dist': s_dist,
        'b_df': b_df,
        's_df': s_df
    }

# Execute domain calculations
year_results = {}
author_results = {}
journal_results = {}
mesh_results = {}

for b_code, s_code, label in COMPARISONS:
    comp_key = f"{b_code}_vs_{s_code}"
    year_results[comp_key] = analyze_year(b_code, s_code, label)
    author_results[comp_key] = analyze_author(b_code, s_code, label)
    journal_results[comp_key] = analyze_journal(b_code, s_code, label)
    mesh_results[comp_key] = analyze_mesh(b_code, s_code, label)

# =============================================================================
# SECTION 4: BUILD COMPREHENSIVE TABLES & ASSESSMENT
# =============================================================================

print("\n📋 Creating Comprehensive Results Table...")
comprehensive_rows = []

domain_dict = {
    'Year': year_results,
    'Author': author_results,
    'Journal': journal_results,
    'MeSH': mesh_results
}

for domain_name, res_dict in domain_dict.items():
    for comp_key, res in res_dict.items():
        comprehensive_rows.append({
            'Analysis': domain_name,
            'Comparison': res['comparison'],
            'Baseline_Count': res['base_count'],
            'Subset_Count': res['sub_count'],
            'Count_Coverage_%': f"{res['count_cov']:.1f}%",
            'Baseline_Total': res['base_total'],
            'Subset_Total': res['sub_total'],
            'Total_Coverage_%': f"{res['total_cov']:.1f}%",
            'Cramers_V': f"{res['cramers_v']:.4f}",
            'Effect_Size': res['interp'],
            'Max_Pct_Diff_%': f"{res['pct_diff']['max_diff']:.2f}%",
            'Mean_Pct_Diff_%': f"{res['pct_diff']['mean_diff']:.2f}%"
        })

comprehensive_df = pd.DataFrame(comprehensive_rows)
print(comprehensive_df.to_string(index=False))

comp_csv_path = os.path.join(OUTPUT_DIR, 'representativeness_comprehensive_results.csv')
comprehensive_df.to_csv(comp_csv_path, index=False)
print(f"💾 Saved: {comp_csv_path}")

# Final Assessment Table
final_rows = []
for _, row in comprehensive_df.iterrows():
    eff = row['Effect_Size']
    if "Very Small" in eff:
        assessment = "✅ Highly Representative"
        score = "Excellent"
    elif "Small" in eff:
        assessment = "✅ Representative"
        score = "Good"
    elif "Medium" in eff:
        assessment = "⚠️ Moderately Representative"
        score = "Acceptable"
    else:
        assessment = "❌ Not Representative"
        score = "Poor"

    final_rows.append({
        'Analysis': row['Analysis'],
        'Comparison': row['Comparison'],
        'Count_Coverage': row['Count_Coverage_%'],
        'Total_Coverage': row['Total_Coverage_%'],
        'Cramers_V': row['Cramers_V'],
        'Max_Pct_Diff': row['Max_Pct_Diff_%'],
        'Assessment': assessment,
        'Score': score
    })

final_df = pd.DataFrame(final_rows)
final_csv_path = os.path.join(OUTPUT_DIR, 'final_representativeness_assessment.csv')
final_df.to_csv(final_csv_path, index=False)
print(f"💾 Saved: {final_csv_path}")

# =============================================================================
# SECTION 5: GENERATE ALL 6 VISUALIZATIONS
# =============================================================================

print("\n🎨 Generating Visualization Panels...")
colors_list = [dataset_colors[DATASET_NAMES[c]] for c in overview_df['Dataset_Code']]

# --- Figure 1: Dataset Overview ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.bar(overview_df['Dataset'], overview_df['Unique_Studies'], color=colors_list)
ax1.set_title('Number of Unique Studies by Dataset', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Studies')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(overview_df['Unique_Studies']):
    ax1.text(i, v + max(overview_df['Unique_Studies'])*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')

ax2.bar(overview_df['Dataset'], overview_df['Unique_Countries'], color=colors_list)
ax2.set_title('Number of Unique Countries by Dataset', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Countries')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(overview_df['Unique_Countries']):
    ax2.text(i, v + max(overview_df['Unique_Countries'])*0.01, str(v), ha='center', va='bottom', fontweight='bold')

ax3.bar(overview_df['Dataset'], overview_df['Unique_Journals'], color=colors_list)
ax3.set_title('Number of Unique Journals by Dataset', fontsize=14, fontweight='bold')
ax3.set_ylabel('Number of Journals')
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(overview_df['Unique_Journals']):
    ax3.text(i, v + max(overview_df['Unique_Journals'])*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_overview_complete.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 2: Year Analysis (3x2 grid) ---
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))

year_counts = overview_df['Unique_Years'].values
ax1.bar(overview_df['Dataset'], year_counts, color=colors_list)
ax1.set_title('Unique Years Count by Dataset', fontweight='bold')
ax1.set_ylabel('Number of Unique Years')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(year_counts):
    ax1.text(i, v + max(year_counts) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')

comp_names = [label for _, _, label in COMPARISONS]
year_covs = [year_results[f"{b}_vs_{s}"]['count_cov'] for b, s, _ in COMPARISONS]
comp_colors = [COLORS['primary_blue'], COLORS['danger_red'], COLORS['dark_green'], COLORS['warning_orange']]

ax2.bar(range(len(comp_names)), year_covs, color=comp_colors)
ax2.set_title('Year Coverage % by Comparison', fontweight='bold')
ax2.set_ylabel('Coverage Percentage (%)')
ax2.set_xticks(range(len(comp_names)))
ax2.set_xticklabels(comp_names, rotation=45, ha='right')
for i, v in enumerate(year_covs):
    ax2.text(i, v + max(year_covs) * 0.01, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# 4 Line charts for each comparison
line_axes = [ax3, ax4, ax5, ax6]
for idx, (b_code, s_code, label) in enumerate(COMPARISONS):
    ax = line_axes[idx]
    res = year_results[f"{b_code}_vs_{s_code}"]
    b_dist = res['b_dist']
    s_dist = res['s_dist']
    comm_yrs = sorted(set(b_dist.index) & set(s_dist.index))
    b_alg = b_dist.reindex(comm_yrs, fill_value=0)
    s_alg = s_dist.reindex(comm_yrs, fill_value=0)

    b_name = DATASET_NAMES[b_code]
    s_name = DATASET_NAMES[s_code]

    ax.plot(comm_yrs, b_alg.values, 'o-', label=b_name, color=dataset_colors[b_name], linewidth=2, markersize=4)
    ax.plot(comm_yrs, s_alg.values, 's-', label=s_name, color=dataset_colors[s_name], linewidth=2, markersize=4)
    ax.set_title(f'Year Distribution: {label}', fontweight='bold')
    ax.set_ylabel('Number of Studies')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'year_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 3: Author Analysis (2x3 grid) ---
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

country_counts = overview_df['Unique_Countries'].values
ax1.bar(overview_df['Dataset'], country_counts, color=colors_list)
ax1.set_title('Unique Countries Count by Dataset', fontweight='bold')
ax1.set_ylabel('Number of Unique Countries')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(country_counts):
    ax1.text(i, v + max(country_counts) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')

author_covs = [author_results[f"{b}_vs_{s}"]['count_cov'] for b, s, _ in COMPARISONS]
ax2.bar(range(len(comp_names)), author_covs, color=comp_colors)
ax2.set_title('Country Coverage % by Comparison', fontweight='bold')
ax2.set_ylabel('Coverage Percentage (%)')
ax2.set_xticks(range(len(comp_names)))
ax2.set_xticklabels(comp_names, rotation=45, ha='right')
for i, v in enumerate(author_covs):
    ax2.text(i, v + max(author_covs) * 0.01, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

bar_axes = [ax3, ax4, ax5, ax6]
for idx, (b_code, s_code, label) in enumerate(COMPARISONS):
    ax = bar_axes[idx]
    res = author_results[f"{b_code}_vs_{s_code}"]
    b_top10 = res['b_dist'].head(10)
    s_top10 = res['s_dist'].reindex(b_top10.index, fill_value=0)

    x = np.arange(len(b_top10))
    width = 0.35
    b_name = DATASET_NAMES[b_code]
    s_name = DATASET_NAMES[s_code]

    ax.bar(x - width / 2, b_top10.values, width, label=b_name, color=dataset_colors[b_name], alpha=0.8)
    ax.bar(x + width / 2, s_top10.values, width, label=s_name, color=dataset_colors[s_name], alpha=0.8)
    ax.set_title(f'Top Countries: {label}', fontweight='bold')
    ax.set_ylabel('Author Affiliations')
    ax.set_xticks(x)
    ax.set_xticklabels(b_top10.index, rotation=45, ha='right')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'author_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 4: Journal Analysis (2x3 grid) ---
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

journal_counts = overview_df['Unique_Journals'].values
ax1.bar(overview_df['Dataset'], journal_counts, color=colors_list)
ax1.set_title('Unique Journals Count by Dataset', fontweight='bold')
ax1.set_ylabel('Number of Unique Journals')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(journal_counts):
    ax1.text(i, v + max(journal_counts) * 0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')

journal_covs = [journal_results[f"{b}_vs_{s}"]['count_cov'] for b, s, _ in COMPARISONS]
ax2.bar(range(len(comp_names)), journal_covs, color=comp_colors)
ax2.set_title('Journal Coverage % by Comparison', fontweight='bold')
ax2.set_ylabel('Coverage Percentage (%)')
ax2.set_xticks(range(len(comp_names)))
ax2.set_xticklabels(comp_names, rotation=45, ha='right')
for i, v in enumerate(journal_covs):
    ax2.text(i, v + max(journal_covs) * 0.01, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

bar_axes = [ax3, ax4, ax5, ax6]
for idx, (b_code, s_code, label) in enumerate(COMPARISONS):
    ax = bar_axes[idx]
    res = journal_results[f"{b_code}_vs_{s_code}"]
    b_cats = res['b_dist'].sort_index()
    s_cats = res['s_dist'].reindex(b_cats.index, fill_value=0)

    x = np.arange(len(b_cats))
    width = 0.35
    b_name = DATASET_NAMES[b_code]
    s_name = DATASET_NAMES[s_code]

    ax.bar(x - width / 2, b_cats.values, width, label=b_name, color=dataset_colors[b_name], alpha=0.8)
    ax.bar(x + width / 2, s_cats.values, width, label=s_name, color=dataset_colors[s_name], alpha=0.8)
    ax.set_title(f'Journal Categories: {label}', fontweight='bold')
    ax.set_ylabel('Publications')
    ax.set_xticks(x)
    ax.set_xticklabels(b_cats.index, rotation=45, ha='right')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'journal_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 5: MeSH Analysis (2x3 grid) ---
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

mesh_covs = [mesh_results[f"{b}_vs_{s}"]['count_cov'] for b, s, _ in COMPARISONS]
ax1.bar(range(len(comp_names)), mesh_covs, color=comp_colors)
ax1.set_title('MeSH Terms Coverage % by Comparison', fontweight='bold')
ax1.set_ylabel('Coverage Percentage (%)')
ax1.set_xticks(range(len(comp_names)))
ax1.set_xticklabels(comp_names, rotation=45, ha='right')
for i, v in enumerate(mesh_covs):
    ax1.text(i, v + max(mesh_covs) * 0.01, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

mesh_counts = overview_df['Unique_MeSH'].values
ax2.bar(overview_df['Dataset'], mesh_counts, color=colors_list)
ax2.set_title('Unique MeSH Terms Count by Dataset', fontweight='bold')
ax2.set_ylabel('Number of Unique MeSH Terms')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(mesh_counts):
    ax2.text(i, v + max(mesh_counts) * 0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')

bar_axes = [ax3, ax4, ax5, ax6]
for idx, (b_code, s_code, label) in enumerate(COMPARISONS):
    ax = bar_axes[idx]
    res = mesh_results[f"{b_code}_vs_{s_code}"]
    b_df = res['b_df']
    s_df = res['s_df']
    b_pmid_col = 'PMID' if 'PMID' in b_df.columns else 'pmid'
    s_pmid_col = 'PMID' if 'PMID' in s_df.columns else 'pmid'

    b_tree_first = b_df.groupby(b_df['treenumber'].str[0])[b_pmid_col].nunique().dropna()
    s_tree_first = s_df.groupby(s_df['treenumber'].str[0])[s_pmid_col].nunique().dropna()

    all_letters = sorted(set(b_tree_first.index) | set(s_tree_first.index))
    b_alg = b_tree_first.reindex(all_letters, fill_value=0)
    s_alg = s_tree_first.reindex(all_letters, fill_value=0)

    x = np.arange(len(all_letters))
    width = 0.35
    b_name = DATASET_NAMES[b_code]
    s_name = DATASET_NAMES[s_code]

    ax.bar(x - width / 2, b_alg.values, width, label=b_name, color=dataset_colors[b_name], alpha=0.8)
    ax.bar(x + width / 2, s_alg.values, width, label=s_name, color=dataset_colors[s_name], alpha=0.8)
    ax.set_title(f'MeSH Categories: {label}', fontweight='bold')
    ax.set_ylabel('Number of Studies')
    ax.set_xticks(x)
    ax.set_xticklabels(all_letters, rotation=0)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mesh_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 6: Heatmap Summaries ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
analyses = ['Year', 'Author', 'Journal', 'MeSH']

count_cov_mat = np.zeros((len(analyses), len(comp_names)))
cramers_mat = np.zeros((len(analyses), len(comp_names)))
max_diff_mat = np.zeros((len(analyses), len(comp_names)))

for i, analysis in enumerate(analyses):
    for j, comp_label in enumerate(comp_names):
        sub = comprehensive_df[(comprehensive_df['Analysis'] == analysis) & (comprehensive_df['Comparison'] == comp_label)]
        if not sub.empty:
            count_cov_mat[i, j] = float(sub.iloc[0]['Count_Coverage_%'].replace('%', ''))
            cramers_mat[i, j] = float(sub.iloc[0]['Cramers_V'])
            max_diff_mat[i, j] = float(sub.iloc[0]['Max_Pct_Diff_%'].replace('%', ''))

im1 = ax1.imshow(count_cov_mat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax1.set_title('Count Coverage % by Analysis Type', fontweight='bold')
ax1.set_xticks(range(len(comp_names)))
ax1.set_yticks(range(len(analyses)))
ax1.set_xticklabels(comp_names, rotation=45, ha='right')
ax1.set_yticklabels(analyses)
for i in range(len(analyses)):
    for j in range(len(comp_names)):
        ax1.text(j, i, f'{count_cov_mat[i, j]:.1f}%', ha="center", va="center", fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.6)

im2 = ax2.imshow(cramers_mat, cmap='RdYlGn_r', vmin=0, vmax=0.3, aspect='auto')
ax2.set_title('Cramér\'s V Effect Sizes', fontweight='bold')
ax2.set_xticks(range(len(comp_names)))
ax2.set_yticks(range(len(analyses)))
ax2.set_xticklabels(comp_names, rotation=45, ha='right')
ax2.set_yticklabels(analyses)
for i in range(len(analyses)):
    for j in range(len(comp_names)):
        ax2.text(j, i, f'{cramers_mat[i, j]:.4f}', ha="center", va="center", fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.6)

im3 = ax3.imshow(max_diff_mat, cmap='RdYlGn_r', vmin=0, vmax=6, aspect='auto')
ax3.set_title('Maximum Percentage Differences', fontweight='bold')
ax3.set_xticks(range(len(comp_names)))
ax3.set_yticks(range(len(analyses)))
ax3.set_xticklabels(comp_names, rotation=45, ha='right')
ax3.set_yticklabels(analyses)
for i in range(len(analyses)):
    for j in range(len(comp_names)):
        ax3.text(j, i, f'{max_diff_mat[i, j]:.2f}%', ha="center", va="center", fontweight='bold')
plt.colorbar(im3, ax=ax3, shrink=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_representativeness_heatmaps.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n🎉 ALL OUTPUTS GENERATED SUCCESSFULLY IN public/representativeness/!")
