import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

csv_files = list(RAW_DIR.glob("*.csv"))

print("=" * 60)
print("E-COMMERCE DATASET PROFILING")
print("=" * 60)

for file in csv_files:
    try:
        df = pd.read_csv(file)

        print(f"\n📁 {file.name}")
        print(f"Shape       : {df.shape}")
        print(f"Duplicates  : {df.duplicated().sum()}")
        print(f"Missing vals: {df.isnull().sum().sum()}")

        print("\nColumns:")
        print(list(df.columns))

        print("\nData types:")
        print(df.dtypes)

        print("-" * 60)

    except Exception as e:
        print(f"❌ Error reading {file.name}: {e}")