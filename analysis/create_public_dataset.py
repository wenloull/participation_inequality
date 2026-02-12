import pandas as pd
import numpy as np
import os

def create_aggregated_dataset(size_suffix):
    """
    Creates an aggregated dataset (Country-Disease-Year) from raw files.
    size_suffix: '138k' or '70k'
    """
    print(f"\nProcessing {size_suffix} dataset...")
    
    # Define file paths
    base_dir = r"c:/Users/dell/PycharmProjects/nlp2/participation_inequality/analysis"
    pmid_file = os.path.join(base_dir, f"pmid_cause_{size_suffix}.csv")
    year_file = os.path.join(base_dir, f"year_{size_suffix}.csv")
    geoinfor_file = os.path.join(base_dir, "geoinfor.csv")
    disease_mapping_file = os.path.join(base_dir, "disease_mapping.csv")
    
    # Check if files exist
    for f in [pmid_file, year_file, geoinfor_file, disease_mapping_file]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return None

    # Load data
    print("  Loading CSVs...")
    try:
        pmid_df = pd.read_csv(pmid_file)
        year_df = pd.read_csv(year_file)
        geoinfor_df = pd.read_csv(geoinfor_file)
        disease_map = pd.read_csv(disease_mapping_file)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    # Filter valid years (same as in original scripts)
    year_df = year_df[(year_df['YEAR'] >= 2000) & (year_df['YEAR'] <= 2024)]

    # 1. Merge PMID + Cause
    # Note: pmid_cause has 'cause_id', map it to 'REI Name' (Disease)
    # disease_map has 'REI ID' and 'REI Name'
    print("  Mapping diseases...")
    pmid_disease = pmid_df.merge(disease_map[['REI ID', 'REI Name']], 
                                left_on='cause_id', right_on='REI ID', how='left')
    
    # 2. Merge with Year
    print("  Merging with Year...")
    # Both have PMID
    pmid_disease_year = pmid_disease.merge(year_df[['PMID', 'YEAR']], on='PMID', how='inner')
    
    # 3. Merge with Geoinfor (Participants per country)
    print("  Merging with Geoinfor...")
    # geoinfor has PMID, ISO3, Amount
    full_df = pmid_disease_year.merge(geoinfor_df[['PMID', 'ISO3', 'Amount']], on='PMID', how='inner')
    
    # 4. Aggregate
    print("  Aggregating...")
    # Group by ISO3, Disease (REI Name), YEAR
    # Sum 'Amount' to get Total Participants
    # Count unique 'PMID' to get Study Count
    aggregated_df = full_df.groupby(['ISO3', 'REI Name', 'YEAR']).agg({
        'Amount': 'sum',
        'PMID': 'nunique'
    }).reset_index()
    
    aggregated_df.rename(columns={
        'REI Name': 'Disease',
        'Amount': 'Total_Participants',
        'PMID': 'Total_Studies'
    }, inplace=True)
    
    # 5. Save
    output_file = os.path.join(base_dir, f"public_aggregated_participants_{size_suffix}.csv")
    aggregated_df.to_csv(output_file, index=False)
    print(f"  Saved aggregated dataset to: {output_file}")
    print(f"  Rows: {len(aggregated_df)}")
    print(f"  Columns: {list(aggregated_df.columns)}")
    
    return output_file

def main():
    print("Starting Open Science Dataset Creation...")
    
    # Create for 138k (Main analysis)
    create_aggregated_dataset('138k')
    
    # Create for 70k (Clinical Trials / Specific analysis)
    create_aggregated_dataset('70k')
    
    print("\nDone.")

if __name__ == "__main__":
    main()
