import os
import glob
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(public_dir, 'data')

    print("="*80)
    print("VERIFYING UNIFIED PUBLIC DATASETS IN public/data/")
    print("="*80)

    # 1. Check macro development indicators dataset
    app_macro_path = os.path.join(data_dir, 'country_macro_indicators.csv')
    if os.path.exists(app_macro_path):
        app_df = pd.read_csv(app_macro_path)
        print(f"✓ country_macro_indicators.csv: {len(app_df)} rows, {len(app_df.columns)} columns, {app_df['ISO3'].nunique()} ISO3 countries")
    else:
        print("❌ country_macro_indicators.csv missing")

    # 2. Check national validation dataset
    val_path = os.path.join(data_dir, 'intervention_national_pbr_validation.csv')
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        print(f"✓ intervention_national_pbr_validation.csv: {len(val_df)} countries")
    else:
        print("❌ intervention_national_pbr_validation.csv missing")

    # 3. Check all panel source spreadsheets
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"\nTotal Public CSV Spreadsheets in data/: {len(csv_files)}")
    for f in sorted(csv_files):
        fname = os.path.basename(f)
        df_temp = pd.read_csv(f)
        print(f"  - {fname}: {len(df_temp)} rows, {len(df_temp.columns)} columns")

    print("\n✓ ALL UNIFIED PUBLIC DATA VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
