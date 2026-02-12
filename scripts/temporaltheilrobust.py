import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

# Your existing color scheme
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


def robust_temporal_theil_analysis(temporal_pbr_data, custom_diseases, n_bootstrap=200):
    """
    Robust temporal Theil analysis with bootstrapped confidence intervals
    """
    print("ROBUST TEMPORAL THEIL ANALYSIS WITH BOOTSTRAP CIs")

    if temporal_pbr_data is None:
        return None

    # Create 2-year periods
    temporal_pbr_data['Period'] = (temporal_pbr_data['YEAR'] // 2) * 2
    periods = sorted(temporal_pbr_data['Period'].unique())

    results = []
    bootstrap_results = []

    for period in periods:
        print(f"   Processing {period}-{period + 1}...")
        period_data = temporal_pbr_data[temporal_pbr_data['Period'] == period]

        # Aggregate by Country-Disease for the period
        period_agg = period_data.groupby(['ISO3', 'Disease']).agg({
            'Total_Participants': 'sum',
            'Avg_DALYs': 'mean'
        }).reset_index()

        period_agg['Period_PBR'] = period_agg['Total_Participants'] / period_agg['Avg_DALYs']
        period_agg = period_agg[~np.isinf(period_agg['Period_PBR'])]
        period_agg = period_agg[period_agg['Period_PBR'] > 0]

        if len(period_agg) < 30:  # Increased minimum threshold
            print(f"     Skipping {period}: insufficient data (n={len(period_agg)})")
            continue

        # Main Theil calculation
        theil_result = calculate_theil_single_period(period_agg, custom_diseases)

        if theil_result is not None:
            results.append({
                'Period': period,
                'Between_Percent': theil_result['between_percent'],
                'Within_Percent': theil_result['within_percent'],
                'Total_Theil': theil_result['total_theil'],
                'Sample_Size': len(period_agg),
                'Diseases_Count': period_agg['Disease'].nunique(),
                'Countries_Count': period_agg['ISO3'].nunique()
            })

            # Bootstrap for confidence intervals
            bootstrap_between = []
            bootstrap_within = []

            for boot_i in range(n_bootstrap):
                if boot_i % 50 == 0:
                    print(f"     Bootstrap progress: {boot_i}/{n_bootstrap}")

                # Bootstrap sample (resample countries with replacement)
                unique_countries = period_agg['ISO3'].unique()
                boot_countries = np.random.choice(unique_countries,
                                                  size=len(unique_countries),
                                                  replace=True)

                boot_data = []
                for country in boot_countries:
                    country_data = period_agg[period_agg['ISO3'] == country]
                    boot_data.append(country_data)

                if len(boot_data) > 0:
                    boot_period_agg = pd.concat(boot_data, ignore_index=True)
                    boot_theil = calculate_theil_single_period(boot_period_agg, custom_diseases)

                    if boot_theil is not None:
                        bootstrap_between.append(boot_theil['between_percent'])
                        bootstrap_within.append(boot_theil['within_percent'])

            # Calculate confidence intervals
            if len(bootstrap_between) > 10:
                between_ci_lower = np.percentile(bootstrap_between, 2.5)
                between_ci_upper = np.percentile(bootstrap_between, 97.5)
                within_ci_lower = np.percentile(bootstrap_within, 2.5)
                within_ci_upper = np.percentile(bootstrap_within, 97.5)

                bootstrap_results.append({
                    'Period': period,
                    'Between_CI_Lower': between_ci_lower,
                    'Between_CI_Upper': between_ci_upper,
                    'Within_CI_Lower': within_ci_lower,
                    'Within_CI_Upper': within_ci_upper,
                    'Bootstrap_N': len(bootstrap_between)
                })

    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    bootstrap_df = pd.DataFrame(bootstrap_results)

    # Merge results with bootstrap CIs
    if len(bootstrap_df) > 0:
        final_results = results_df.merge(bootstrap_df, on='Period', how='left')
    else:
        final_results = results_df
        print("Warning: No bootstrap results available")

    return final_results


def calculate_theil_single_period(period_agg, custom_diseases):
    """
    Calculate Theil decomposition for a single period
    """
    # Filter to custom diseases only
    period_agg = period_agg[period_agg['Disease'].isin(custom_diseases)]

    if len(period_agg) < 10:
        return None

    # Calculate disease means
    disease_means = period_agg.groupby('Disease')['Period_PBR'].agg(['mean', 'count']).reset_index()
    disease_means.columns = ['Disease', 'Disease_Mean_PBR', 'Disease_Count']

    # Overall mean
    overall_mean = period_agg['Period_PBR'].mean()

    # Theil components
    between_component = 0
    within_component = 0
    total_n = len(period_agg)

    for _, disease_row in disease_means.iterrows():
        disease = disease_row['Disease']
        disease_mean = disease_row['Disease_Mean_PBR']
        disease_count = disease_row['Disease_Count']

        # Between-disease component
        if disease_mean > 0 and overall_mean > 0:
            between_component += (disease_count / total_n) * (disease_mean / overall_mean) * np.log(
                disease_mean / overall_mean)

        # Within-disease component
        disease_data = period_agg[period_agg['Disease'] == disease]
        for _, country_row in disease_data.iterrows():
            country_pbr = country_row['Period_PBR']
            if country_pbr > 0 and disease_mean > 0:
                within_component += (1 / total_n) * (country_pbr / overall_mean) * np.log(country_pbr / disease_mean)

    total_theil = between_component + within_component

    if total_theil > 0:
        between_pct = (between_component / total_theil) * 100
        within_pct = (within_component / total_theil) * 100
    else:
        between_pct = within_pct = 0

    return {
        'total_theil': total_theil,
        'between_component': between_component,
        'within_component': within_component,
        'between_percent': between_pct,
        'within_percent': within_pct
    }


def calculate_gini_coefficient(values):
    """Calculate Gini coefficient"""
    values = np.array(values)
    values = values[values > 0]
    if len(values) <= 1:
        return 0

    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)

    if cumsum[-1] == 0:
        return 0

    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def lorenz_curve_data(values):
    """Calculate Lorenz curve data points"""
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])

    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    total_sum = cumsum[-1]

    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0], cumsum / total_sum])

    return x, y


def calculate_temporal_lorenz_data(temporal_pbr_data, years=[2000, 2008, 2019, 2021]):
    """Calculate Lorenz curves for specific single years"""
    lorenz_data = {}

    for year in years:
        # Get data for THIS SPECIFIC YEAR ONLY
        year_data = temporal_pbr_data[temporal_pbr_data['YEAR'] == year]

        if len(year_data) == 0:
            print(f"Warning: No data found for year {year}")
            continue

        if len(year_data) > 0:
            # Aggregate by Country-Disease for THIS SINGLE YEAR
            year_agg = year_data.groupby(['ISO3', 'Disease']).agg({
                'Total_Participants': 'sum',
                'Avg_DALYs': 'mean'
            }).reset_index()

            year_agg['PBR'] = year_agg['Total_Participants'] / year_agg['Avg_DALYs']
            year_agg = year_agg[~np.isinf(year_agg['PBR'])]
            year_agg = year_agg[year_agg['PBR'] > 0]

            if len(year_agg) > 10:
                x, y = lorenz_curve_data(year_agg['PBR'].values)
                gini = calculate_gini_coefficient(year_agg['PBR'].values)
                lorenz_data[year] = {'x': x, 'y': y, 'gini': gini, 'n': len(year_agg)}
                print(f"Year {year}: {len(year_agg)} observations, Gini = {gini:.3f}")
            else:
                print(f"Year {year}: insufficient data ({len(year_agg)} observations)")

    return lorenz_data


def segmented_regression_analysis(temporal_results):
    """
    Test for structural breaks in the temporal trend
    """
    print("SEGMENTED REGRESSION ANALYSIS")

    if len(temporal_results) < 6:  # Need sufficient data for segmented regression
        print("Insufficient data for segmented regression")
        return None

    periods = temporal_results['Period'].values
    between_pct = temporal_results['Between_Percent'].values

    # Test potential breakpoints
    breakpoint_results = []

    for potential_break in periods[2:-2]:  # Exclude first and last 2 periods
        # Split data
        pre_break = temporal_results[temporal_results['Period'] <= potential_break]
        post_break = temporal_results[temporal_results['Period'] > potential_break]

        if len(pre_break) >= 3 and len(post_break) >= 3:
            # Linear regression for each segment
            slope_pre, intercept_pre, r_pre, p_pre, se_pre = stats.linregress(
                pre_break['Period'], pre_break['Between_Percent'])
            slope_post, intercept_post, r_post, p_post, se_post = stats.linregress(
                post_break['Period'], post_break['Between_Percent'])

            # Calculate improvement in fit
            # Single regression R²
            slope_full, _, r_full, _, _ = stats.linregress(periods, between_pct)
            r2_full = r_full ** 2

            # Segmented R² (weighted by sample size)
            n_pre, n_post = len(pre_break), len(post_break)
            r2_segmented = (n_pre * r_pre ** 2 + n_post * r_post ** 2) / (n_pre + n_post)

            breakpoint_results.append({
                'Breakpoint': potential_break,
                'Slope_Pre': slope_pre,
                'Slope_Post': slope_post,
                'P_Pre': p_pre,
                'P_Post': p_post,
                'R2_Improvement': r2_segmented - r2_full,
                'N_Pre': n_pre,
                'N_Post': n_post
            })

    if len(breakpoint_results) > 0:
        breakpoint_df = pd.DataFrame(breakpoint_results)
        best_break = breakpoint_df.loc[breakpoint_df['R2_Improvement'].idxmax()]

        print(f"Best breakpoint: {best_break['Breakpoint']}")
        print(f"Pre-break slope: {best_break['Slope_Pre']:.3f}%/period (p={best_break['P_Pre']:.3f})")
        print(f"Post-break slope: {best_break['Slope_Post']:.3f}%/period (p={best_break['P_Post']:.3f})")
        print(f"R² improvement: {best_break['R2_Improvement']:.3f}")

        return breakpoint_df, best_break

    return None, None


def create_nature_quality_temporal_plot(temporal_results, temporal_pbr_data, breakpoint_analysis=None, decomposition_results=None):
    """
    Create 1x3 publication-quality temporal plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Temporal Evolution of Clinical Trial Inequality Structure (2000-2024)',
                 fontsize=18, fontweight='bold')

    # Panel A: Lorenz curves for specific years
    if temporal_pbr_data is not None:
        lorenz_years = [2000, 2008, 2019, 2021]
        lorenz_data = calculate_temporal_lorenz_data(temporal_pbr_data, lorenz_years)

        colors = [COLORS['primary_blue'], COLORS['accent_orange'], COLORS['primary_red'], COLORS['neutral_gray']]

        # Plot equality line
        ax1.plot([0, 1], [0, 1], '--', color='black', alpha=0.5, linewidth=2, label='Perfect Equality')

        for i, (year, data) in enumerate(lorenz_data.items()):
            if i < len(colors):
                ax1.plot(data['x'], data['y'], '-', color=colors[i], linewidth=3,
                         label=f"{year} (Gini={data['gini']:.3f})")

        ax1.set_xlabel('Cumulative Share of Country-Disease Pairs')
        ax1.set_ylabel('Cumulative Share of Participants')
        ax1.set_title('(A) Lorenz Curves by Year')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'Temporal data\nnot available', transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12)
        ax1.set_title('(A) Lorenz Curves by Year')

    # Panel B: Gini coefficient trends with/without top 3 drivers
    if temporal_pbr_data is not None and decomposition_results is not None:
        # Get top 3 driver diseases from your main analysis
        top_3_drivers = decomposition_results.head(3)['Disease'].tolist()

        temporal_pbr_data['Period'] = (temporal_pbr_data['YEAR'] // 2) * 2
        periods = sorted(temporal_pbr_data['Period'].unique())

        gini_all_results = []
        gini_reduced_results = []

        for period in periods:
            period_data = temporal_pbr_data[temporal_pbr_data['Period'] == period]

            # All diseases
            period_agg_all = period_data.groupby(['ISO3', 'Disease']).agg({
                'Total_Participants': 'sum',
                'Avg_DALYs': 'mean'
            }).reset_index()

            period_agg_all['Period_PBR'] = period_agg_all['Total_Participants'] / period_agg_all['Avg_DALYs']
            period_agg_all = period_agg_all[~np.isinf(period_agg_all['Period_PBR'])]
            period_agg_all = period_agg_all[period_agg_all['Period_PBR'] > 0]

            if len(period_agg_all) > 10:
                gini_all = calculate_gini_coefficient(period_agg_all['Period_PBR'].values)
                gini_all_results.append({'Period': period, 'Gini': gini_all * 100})

            # Without top 3 drivers
            period_agg_reduced = period_agg_all[~period_agg_all['Disease'].isin(top_3_drivers)]

            if len(period_agg_reduced) > 10:
                gini_reduced = calculate_gini_coefficient(period_agg_reduced['Period_PBR'].values)
                gini_reduced_results.append({'Period': period, 'Gini': gini_reduced * 100})

        if len(gini_all_results) > 0 and len(gini_reduced_results) > 0:
            gini_all_df = pd.DataFrame(gini_all_results)
            gini_reduced_df = pd.DataFrame(gini_reduced_results)

            # Plot both lines
            ax2.scatter(gini_all_df['Period'], gini_all_df['Gini'], color=COLORS['danger_red'],
                        s=80, alpha=0.7, zorder=3, label='All Diseases')
            ax2.scatter(gini_reduced_df['Period'], gini_reduced_df['Gini'], color=COLORS['dark_green'],
                        s=80, alpha=0.7, zorder=3, label='Remove Top 3 Diseases')

            # Smooth trends for both
            period_smooth = np.linspace(min(gini_all_df['Period'].min(), gini_reduced_df['Period'].min()),
                                        max(gini_all_df['Period'].max(), gini_reduced_df['Period'].max()), 100)

            # All diseases trend
            if len(gini_all_df) >= 4:
                try:
                    spline_all = UnivariateSpline(gini_all_df['Period'], gini_all_df['Gini'], s=len(gini_all_df) * 2)
                    smooth_all = spline_all(period_smooth)
                    ax2.plot(period_smooth, smooth_all, '-', color=COLORS['danger_red'],
                             linewidth=3, alpha=0.8)
                except:
                    pass

            # Linear trend for all diseases
            slope_all, intercept_all, r_all, p_all, _ = stats.linregress(gini_all_df['Period'], gini_all_df['Gini'])
            trend_all = slope_all * gini_all_df['Period'] + intercept_all
            ax2.plot(gini_all_df['Period'], trend_all, '--', color=COLORS['danger_red'],
                     linewidth=2, alpha=0.9)

            # Reduced diseases trend
            if len(gini_reduced_df) >= 4:
                try:
                    spline_reduced = UnivariateSpline(gini_reduced_df['Period'], gini_reduced_df['Gini'],
                                                      s=len(gini_reduced_df) * 2)
                    smooth_reduced = spline_reduced(period_smooth)
                    ax2.plot(period_smooth, smooth_reduced, '-', color=COLORS['dark_green'],
                             linewidth=3, alpha=0.8)
                except:
                    pass

            # Linear trend for reduced diseases
            slope_red, intercept_red, r_red, p_red, _ = stats.linregress(gini_reduced_df['Period'],
                                                                         gini_reduced_df['Gini'])
            trend_red = slope_red * gini_reduced_df['Period'] + intercept_red
            ax2.plot(gini_reduced_df['Period'], trend_red, '--', color=COLORS['dark_green'],
                     linewidth=2, alpha=0.9)

            ax2.text(0.02, 0.98,
                     f"All: {slope_all:.3f}%/period (R²={r_all ** 2:.3f})\nReduced: {slope_red:.3f}%/period (R²={r_red ** 2:.3f})",
                     transform=ax2.transAxes, va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        ax2.set_xlabel('Period Start Year')
        ax2.set_ylabel('Gini Coefficient (%)')
        ax2.set_title('(B) Inequality Reduction by Removing Driver Diseases')
        ax2.set_xticks(range(2000, 2025, 4))
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10, loc='lower center',  ncol=2)
        ax2.grid(True, alpha=0.3)

    # Panel C: Combined Between-disease and Within-disease trends
    periods = temporal_results['Period']
    between_pct = temporal_results['Between_Percent']
    within_pct = temporal_results['Within_Percent']
    period_smooth = np.linspace(periods.min(), periods.max(), 100)

    # Plot Between-disease
    ax3.scatter(periods, between_pct, color=COLORS['danger_red'],
                s=80, alpha=0.7, zorder=3, label='Between-Disease (observed)')

    # Add smooth confidence intervals for between-disease
    if 'Between_CI_Lower' in temporal_results.columns:
        ci_lower = temporal_results['Between_CI_Lower']
        ci_upper = temporal_results['Between_CI_Upper']

        if len(periods) >= 4:
            try:
                spline_ci_lower = UnivariateSpline(periods, ci_lower, s=len(periods) * 1.5)
                spline_ci_upper = UnivariateSpline(periods, ci_upper, s=len(periods) * 1.5)
                smooth_ci_lower = spline_ci_lower(period_smooth)
                smooth_ci_upper = spline_ci_upper(period_smooth)

                ax3.fill_between(period_smooth, smooth_ci_lower, smooth_ci_upper,
                                 color=COLORS['danger_red'], alpha=0.2, label='Between 95% CI')
            except:
                ax3.fill_between(periods, ci_lower, ci_upper,
                                 color=COLORS['danger_red'], alpha=0.2, label='Between 95% CI')

    # Smooth trend line for between-disease (SOLID)
    if len(periods) >= 4:
        try:
            spline = UnivariateSpline(periods, between_pct, s=len(periods) * 2)
            smooth_between = spline(period_smooth)
            ax3.plot(period_smooth, smooth_between, '-',
                     color=COLORS['danger_red'], linewidth=3, alpha=0.8, label='Between smooth trend')
        except:
            pass

    # Linear trend line for between-disease (DASHED)
    slope_b, intercept_b, r_value_b, p_value_b, _ = stats.linregress(periods, between_pct)
    trend_line_b = slope_b * periods + intercept_b
    ax3.plot(periods, trend_line_b, '--', color=COLORS['danger_red'],
             linewidth=2, alpha=0.9, label='Between linear trend')

    # Plot Within-disease
    ax3.scatter(periods, within_pct, color=COLORS['dark_green'],
                s=80, alpha=0.7, zorder=3, label='Within-Disease (observed)')

    # Add smooth confidence intervals for within-disease
    if 'Within_CI_Lower' in temporal_results.columns:
        within_ci_lower = temporal_results['Within_CI_Lower']
        within_ci_upper = temporal_results['Within_CI_Upper']

        if len(periods) >= 4:
            try:
                spline_within_ci_lower = UnivariateSpline(periods, within_ci_lower, s=len(periods) * 1.5)
                spline_within_ci_upper = UnivariateSpline(periods, within_ci_upper, s=len(periods) * 1.5)
                smooth_within_ci_lower = spline_within_ci_lower(period_smooth)
                smooth_within_ci_upper = spline_within_ci_upper(period_smooth)

                ax3.fill_between(period_smooth, smooth_within_ci_lower, smooth_within_ci_upper,
                                 color=COLORS['dark_green'], alpha=0.2, label='Within 95% CI')
            except:
                ax3.fill_between(periods, within_ci_lower, within_ci_upper,
                                 color=COLORS['dark_green'], alpha=0.2, label='Within 95% CI')

    # Smooth trend for within-disease (SOLID)
    if len(periods) >= 4:
        try:
            spline_within = UnivariateSpline(periods, within_pct, s=len(periods) * 2)
            smooth_within = spline_within(period_smooth)
            ax3.plot(period_smooth, smooth_within, '-',
                     color=COLORS['dark_green'], linewidth=3, alpha=0.8, label='Within smooth trend')
        except:
            pass

    # Linear trend line for within-disease (DASHED)
    slope_w, intercept_w, r_value_w, p_value_w, _ = stats.linregress(periods, within_pct)
    trend_line_w = slope_w * periods + intercept_w
    ax3.plot(periods, trend_line_w, '--', color=COLORS['dark_green'],
             linewidth=2, alpha=0.9, label='Within linear trend')

    ax3.set_xlabel('Period Start Year')
    ax3.set_ylabel('% of Total Inequality')
    ax3.set_title('(C) Disease-Driven vs Country-Driven Inequality')
    ax3.set_xticks(range(2000, 2025, 4))
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=7, loc='center right', bbox_to_anchor=(1.02, 0.5))
    ax3.grid(True, alpha=0.3)

    # Add statistical annotation
    ax3.text(0.02, 0.98,
             f"Between: {slope_b:.2f}%/period (R²={r_value_b**2:.3f}, p={p_value_b:.3f})\nWithin: {slope_w:.2f}%/period (R²={r_value_w**2:.3f}, p={p_value_w:.3f})",
             transform=ax3.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('nature_temporal_theil_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


# Main execution function
def run_robust_temporal_analysis(temporal_pbr_data, custom_diseases, decomposition_results=None):
    """
    Run the complete robust temporal analysis for Nature submission
    """
    print("RUNNING ROBUST TEMPORAL ANALYSIS FOR NATURE SUBMISSION")
    print("=" * 70)

    # Step 1: Robust Theil analysis with bootstrap CIs
    temporal_results = robust_temporal_theil_analysis(temporal_pbr_data, custom_diseases, n_bootstrap=200)

    if temporal_results is None or len(temporal_results) < 3:
        print("Insufficient temporal data for robust analysis")
        return None

    # Step 2: Segmented regression analysis
    print("\n" + "=" * 70)
    breakpoint_df, best_break = segmented_regression_analysis(temporal_results)

    # Step 3: Create publication-quality visualization
    print("\n" + "=" * 70)
    print("CREATING NATURE-QUALITY VISUALIZATION")
    fig = create_nature_quality_temporal_plot(temporal_results, temporal_pbr_data, (breakpoint_df, best_break), decomposition_results)

    # Step 4: Summary statistics
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("=" * 70)

    periods = temporal_results['Period']
    between_pct = temporal_results['Between_Percent']
    within_pct = temporal_results['Within_Percent']

    print(f"Time span: {periods.min()}-{periods.max()}")
    print(f"Number of periods: {len(temporal_results)}")
    print(f"Between-disease inequality:")
    print(f"  Start: {between_pct.iloc[0]:.1f}% ({periods.iloc[0]})")
    print(f"  End: {between_pct.iloc[-1]:.1f}% ({periods.iloc[-1]})")
    print(f"  Change: {between_pct.iloc[-1] - between_pct.iloc[0]:.1f} percentage points")

    # Statistical significance of trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(periods, between_pct)
    print(f"  Trend: {slope:.3f}%/period (p={p_value:.3f})")
    print(f"  R² = {r_value ** 2:.3f}")

    if p_value < 0.05:
        direction = "decline" if slope < 0 else "increase"
        print(f"  Significant {direction} in disease-driven inequality")
    else:
        print(f"  No significant temporal trend")

    # Save results
    temporal_results.to_csv('robust_temporal_theil_results.csv', index=False)
    if breakpoint_df is not None:
        breakpoint_df.to_csv('segmented_regression_results.csv', index=False)

    print(f"\nFiles saved:")
    print(f"- robust_temporal_theil_results.csv")
    print(f"- nature_temporal_theil_analysis.png")
    if breakpoint_df is not None:
        print(f"- segmented_regression_results.csv")

    return temporal_results, (breakpoint_df, best_break), fig