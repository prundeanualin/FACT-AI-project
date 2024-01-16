
import os
import pandas as pd
from preprocess_datasets import list_of_useful_columns


def perform_nan_analysis(df, columns_to_analyze, top_k_items=10):
    print(f"NaN analysis- percentage of rows with NaN values out of {len(df)} total rows")
    nan_percentage_per_item = dict()
    for col in columns_to_analyze:
        nan_percentage_per_item[col] = df[col].isna().sum() / len(df) * 100
    item_most_nan = {k: v for k, v in sorted(nan_percentage_per_item.items(), key=lambda item: item[1], reverse=True)}

    print(nan_percentage_per_item)
    print(f"\nTop {top_k_items} items with most NaN values")
    print({k: item_most_nan[k] for k in list(item_most_nan)[:top_k_items]})
    print(f"\nTop {top_k_items} items with fewest NaN")
    print({k: item_most_nan[k] for k in list(item_most_nan)[-top_k_items:]})
    print(f"Average percentage of NaN in items")
    print(sum(nan_percentage_per_item.values()) / len(nan_percentage_per_item.values()))

    return nan_percentage_per_item

# Make sure you have the pre-processed dataset in data/pisa2015/dataset.csv
dataset_filepath = 'data/pisa2015/dataset_not_melted.csv'

if not os.path.exists(dataset_filepath):
    raise Exception(f"Make sure you have the pre-processed dataset in {dataset_filepath}")

print("Loading the dataset...")
df = pd.read_csv(dataset_filepath)

print(df.columns.tolist())

# Columns representing the cognitive items(questions) only with the valid score
item_columns = df.columns.difference(list_of_useful_columns)

# Specify datatypes to reduce in-memory size
for col in df.columns.tolist():
    if col == 'CNT':
        df[col] = df[col].astype("category")
    elif col in item_columns:
        df[col] = pd.to_numeric(df[col])

print(df)

perform_nan_analysis(df, item_columns)

final_df_reshaped = pd.melt(df, id_vars=list_of_useful_columns, value_vars=item_columns.tolist(), var_name='item_id', value_name='score')
print("\n\nAfter melting (with NaN item scores) - 313677736")
print(final_df_reshaped.head(15))
print(final_df_reshaped)


print("\nKeeping only the rows with not NaN scores - 27032866 - 8.61 % from all the initial rows")
final_df_reshaped.dropna(subset=['score'], inplace=True)
print(final_df_reshaped.head(30))
print(final_df_reshaped)

final_df_reshaped.to_csv('data/pisa2015/dataset.csv', index=False)
