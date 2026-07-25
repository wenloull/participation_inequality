import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    csv_path = "/Users/wen/Desktop/participation_inequality/public/geoinfor183_disease_matched.csv"
    output_dir = "/Users/wen/Desktop/participation_inequality/public"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path).rename(columns={'PMID': 'pmid', 'Amount': 'amount'})
    
    # Group by country and calculate total enrollment
    country_stats = df.groupby('country').agg(
        total_amount=('amount', 'sum'),
        study_count=('pmid', 'count')
    ).reset_index()
    
    # Sort by total enrollment
    country_stats = country_stats.sort_values(by='total_amount', ascending=False)
    
    # Compute basic stats
    total_countries = len(country_stats)
    total_participants = country_stats['total_amount'].sum()
    mean_enrollment = country_stats['total_amount'].mean()
    median_enrollment = country_stats['total_amount'].median()
    
    print(f"Total unique countries: {total_countries}")
    print(f"Total participants: {total_participants}")
    
    # Save the aggregated statistics
    country_stats.to_csv(os.path.join(output_dir, "country_total_enrollment.csv"), index=False)
    
    # Create the figure with side-by-side panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Panel a: Histogram of country enrollment
    enrollment_vals = country_stats['total_amount'].values
    log_enrollment = np.log10(enrollment_vals[enrollment_vals > 0])
    
    sns.histplot(
        log_enrollment,
        bins=15,
        ax=ax1,
        color='grey',
        edgecolor='black',
        alpha=0.7
    )
    
    # Customize ax1
    ax1.set_title("(a) Frequency distribution of countries by total enrollment", loc='left', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Total Participant Enrollment per Country (log10 scale)", fontsize=11)
    ax1.set_ylabel("Number of Countries", fontsize=11)
    
    # Set x-ticks to display original values
    xticks = np.arange(int(np.floor(log_enrollment.min())), int(np.ceil(log_enrollment.max())) + 1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"$10^{int(x)}$" for x in xticks])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add an annotation of basic stats in the upper left corner of panel a
    stats_text = (
        f"Total Countries: {total_countries}\n"
        f"Total Participants: {format_number(total_participants)}\n"
        f"Mean per Country: {format_number(mean_enrollment)}\n"
        f"Median per Country: {format_number(median_enrollment)}"
    )
    ax1.text(
        0.05, 0.95, stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        fontweight='normal',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgrey', alpha=0.9)
    )
    
    # Panel b: Top 30 countries horizontal bar chart
    top_30 = country_stats.head(30)
    
    # Create horizontal bar chart
    bars = ax2.barh(
        y=np.arange(len(top_30)),
        width=top_30['total_amount'],
        color='grey',
        edgecolor='black',
        alpha=0.7,
        height=0.7
    )
    
    # Customize ax2
    ax2.set_yticks(np.arange(len(top_30)))
    ax2.set_yticklabels(top_30['country'])
    ax2.invert_yaxis()  # top country on top
    
    ax2.set_title("(b) Top 30 countries by total participant recruitment", loc='left', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Total Participant Recruitment", fontsize=11)
    ax2.set_ylabel("Country", fontsize=11)
    
    # Set x-limit to leave space for labels
    max_val = top_30['total_amount'].max()
    ax2.set_xlim(0, max_val * 1.15)
    
    # Format x-axis with standard formatted integers
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Add numerical annotations to the bars
    for bar in bars:
        width = bar.get_width()
        label = f" {format_number(width)}"
        ax2.text(
            width,
            bar.get_y() + bar.get_height()/2,
            label,
            va='center',
            ha='left',
            fontsize=8,
            fontweight='semibold'
        )
        
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figures
    png_out = os.path.join(output_dir, "extended_data_fig1.png")
    pdf_out = os.path.join(output_dir, "extended_data_fig1.pdf")
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    print(f"Figures saved to:\n  - {png_out}\n  - {pdf_out}")

if __name__ == '__main__':
    generate_figure()
