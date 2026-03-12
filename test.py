import pandas as pd

# Load the CSV
df = pd.read_csv("combined_dataset.csv")

# Print rows in original CSV
print("Rows in combined_dataset.csv:", len(df))

# Get one row per dataset_id
sample_df = df.drop_duplicates(subset="dataset_id")

# Save to new CSV
output_file = "one_row_per_dataset_id.csv"
sample_df.to_csv(output_file, index=False)

# Print rows in new CSV
print(f"Rows in {output_file}:", len(sample_df))