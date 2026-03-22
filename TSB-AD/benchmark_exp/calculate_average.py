import pandas as pd
import glob

csv_files = sorted(list(glob.glob("eval/metrics/multi/*.csv")))
for csv_file in csv_files:
    data = pd.read_csv(csv_file)
    if len(data) != 180:
        print(f"Warning: {csv_file} has {len(data)} rows, expected 180.")
        continue

    column_means = data.iloc[:, 2:].mean()

    print("File:", csv_file)
    print("Column-wise averages:")
    print(column_means)