# Make sure your initial PISA data is in:
# data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ
# or in:
# data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ

import numpy as np
import pandas as pd
import os

# Pick SPSS or SAS
file_type = 'SAS'
formats = {'SAS': '.sas7bdat', 'SPSS': '.sav'}

base_path = 'data/pisa2015/'
student_directory = f"PUF_{file_type}_COMBINED_CMB_STU_QQQ/"
cognitive_directory = f"PUF_{file_type}_COMBINED_CMB_STU_COG/"

student_questionnaire_file1 = student_directory + 'cy6_ms_cmb_stu_qqq'
student_questionnaire_file2 = student_directory + 'cy6_ms_cmb_stu_qq2'
cognitive_file = cognitive_directory + 'cy6_ms_cmb_stu_cog'

train_split = 0.7
valid_split = 0.1
attacker_train_split = 0.8

# These are the columns on which we will merge two tables
merge_columns = ["CNT", "CNTSCHID", "CNTSTUID"]

# The variables we actually need are below. Identifiers and sensitive attributes. Below description of variables.
# - "CNTSCHID":  School_id
# - "CNTSTUID":  Student_id
# - "CNT":       Country_code
# - "ST004D01T": Gender
# - "Region":    Region
# - "AGE":       Age,
# - "HOMEPOS":   Home_possessions,
# - "HISCED":    Highest_Education_of_parents_(ISCED) - an aggregate of all edu_mother and edu_father variables
# - "OECD":      Whether_OECD_country

# Other possibly useful variables, not used atm
# "ST005Q01TA": "Edu_mother_highest"
# "ST006Q01TA": "Edu_mother_lvl6"
# "ST006Q02TA": "Edu_mother_lvl5A"
# "ST006Q03TA": "Edu_mother_lvl5B"
# "ST006Q04TA": "Edu_mother_lvl4"
# "ST007Q01TA": "Edu_father_highest"
# "ST008Q01TA": "Edu_father_lvl6"
# "ST008Q02TA": "Edu_father_lvl5A"
# "ST008Q03TA": "Edu_father_lvl5B"
# "ST008Q04TA": "Edu_father_lvl4"
# "PARED": "Index_highest_parental_education"
# These are the only columns that we keep for the student questionnaire
list_of_useful_columns = ["CNTSCHID", "CNTSTUID", "CNT", "ST004D01T", "Region", "AGE", "HOMEPOS", "HISCED", "OECD"]


def load_dataset(path):
    if file_type not in formats:
        raise Exception("Incorrect file type, pick 'SAS' or 'SPSS'.")
    filepath = base_path + path + formats[file_type]
    if not os.path.exists(filepath):
        raise Exception(f"Filepath ({filepath}) does not exist. Make sure the datafile is in the right directory.")
    if file_type == 'SPSS':
        return pd.read_spss(filepath)
    else:
        return pd.read_sas(filepath)


def perform_nan_analysis(df, columns_to_analyze, top_k_items=10):
    print(f"NaN analysis- percentage of rows with NaN values out of {len(df)} total rows")
    nan_percentage_per_item = dict()
    for col in columns_to_analyze:
        nan_percentage_per_item[col] = df[col].isna().sum() / len(df) * 100
    item_most_nan = {k: v for k, v in sorted(nan_percentage_per_item.items(), key=lambda item: item[1], reverse=True)}

    print(nan_percentage_per_item)
    print(f"\nTop {top_k_items} items with most NaN values")
    print({k: item_most_nan[k] for k in list(item_most_nan)[:top_k_items]})
    print(f"\nTop {top_k_items} items with fewest NaN values")
    print({k: item_most_nan[k] for k in list(item_most_nan)[-top_k_items:]})
    print(f"Average percentage of NaN in items")
    print(sum(nan_percentage_per_item.values()) / len(nan_percentage_per_item.values()))

    return nan_percentage_per_item


def reshape_dataset(df, list_of_useful_columns):
    print("Reshaping dataset...")
    # Columns representing the cognitive items(questions) only with the valid score
    item_columns = df.columns.difference(list_of_useful_columns)
    for col in item_columns:
        df[col] = pd.to_numeric(df[col])

    perform_nan_analysis(df, item_columns)

    final_df_reshaped = pd.melt(df, id_vars=list_of_useful_columns, value_vars=item_columns.tolist(),
                                var_name='item_id', value_name='score')
    nr_rows_all = len(final_df_reshaped)
    print(f"\n\nAfter melting (with NaN item scores) - {nr_rows_all}")
    print(final_df_reshaped)

    final_df_reshaped.dropna(subset=['score'], inplace=True)
    nr_rows_not_nan = len(final_df_reshaped)
    print(f"\nKeeping only the rows with not NaN scores - {nr_rows_not_nan} - {nr_rows_not_nan / nr_rows_all * 100} % from all the initial rows")
    print(final_df_reshaped)
    return final_df_reshaped


def postprocess_columns(df):
    print("Postprocessing columns...")
    df.rename(columns={'CNTSTUID': 'user_id'}, inplace=True)
    # Filter for OECD countries to remove rows with 'Not applicable'
    df = df.loc[df['OECD'] < 2]
    df['OECD'] = df['OECD'] == 1
    # Process the gender to have values 0 and 1
    df['ST004D01T'] = df['ST004D01T'] == 1
    # Remove NaN values and process based on whether the highest parents education is above ISCED 1
    df.dropna(subset=['HISCED'], inplace=True)
    df['HISCED'] = df['HISCED'] > 1
    # Process based on whether the number of home possessions (wealth index) is above 0
    df['HOMEPOS'] = df['HOMEPOS'] > 0
    df['CNT'] = df['CNT'].astype(str).map(lambda x: x.lstrip("b\'").rstrip("\'"))
    return df


def preprocess_dataset(random_seed):

    print("Start loading original PISA2015 datasets...")

    # Load and process student questionnaire files
    student_df1 = load_dataset(student_questionnaire_file1)
    student_df2 = load_dataset(student_questionnaire_file2)
    # Merge the 2 student questionnaires by adding only non-duplicate columns from qqq_2
    student_df2_new_cols = student_df2[student_df2.columns.difference(student_df1.drop(columns=merge_columns).columns)]

    # Merge the 2 student questionnaire files and keep only the useful columns
    student_df = pd.merge(student_df1, student_df2_new_cols, on=merge_columns)[list_of_useful_columns]

    print("Final student table with both questionnaires")
    print(student_df.columns)
    print(student_df.head())

    # Load and process cognitive questionnaire file
    cognitive_df = load_dataset(cognitive_file)

    # For the cognitive table, keep only the *merge columns* and the ones that keep the *item grade*, which in the codebook
    # appear either as 'Coded Response' or as 'Scored Response' and end in 'C' or 'S'
    cog_keep_column_idxs = [cognitive_df.columns.get_loc(col) for col in merge_columns]
    idx_start_q = cognitive_df.columns.get_loc('DR219Q01AC')
    idx_end_q = cognitive_df.columns.get_loc('DS626Q04C')
    for i in range(idx_start_q, idx_end_q + 1):
        col_name = cognitive_df.columns[i]
        if str(col_name).endswith(('C', 'S')):
            cog_keep_column_idxs.append(i)
    cognitive_df = cognitive_df.iloc[:, cog_keep_column_idxs]

    print("Cognitive filtered columns: ")
    print(cognitive_df.columns.tolist())

    # Merge the processed student questionnaire with the cognitive one, based on the merge columns
    final_df = pd.merge(student_df, cognitive_df, on=merge_columns)

    print("\n\nFinal student + cognitive table")
    print(final_df.columns.tolist())
    print(final_df.head())
    print(final_df.tail())

    final_df = reshape_dataset(final_df, list_of_useful_columns)
    final_df = postprocess_columns(final_df)

    print("Splitting dataset into train/validation/test sets...")
    print(f"Nr of total items: {len(final_df)}")
    train, validation, test = np.split(final_df.sample(frac=1, random_state=random_seed),
                                     [int(train_split * len(final_df)), int((train_split + valid_split) * len(final_df))])
    print("Length of train set: " + str(len(train)))
    print("Length of valid set: " + str(len(validation)))
    print("Length of test set: " + str(len(test)))

    attacker_train, attacker_test = np.split(final_df.sample(frac=1, random_state=random_seed),
                                              [int(attacker_train_split * len(final_df))])

    print("\n\nWriting the final processed train/validate/test + attacker train/test dataframes to csv...")

    train.to_csv('data/pisa2015/pisa.train.csv', index=False)
    validation.to_csv('data/pisa2015/pisa.validation.csv', index=False)
    test.to_csv('data/pisa2015/pisa.test.csv', index=False)

    attacker_train.to_csv('data/pisa2015/pisa.attacker.train.csv', index=False)
    attacker_test.to_csv('data/pisa2015/pisa.attacker.test.csv', index=False)
    print("Done with PISA dataset preprocessing!")

preprocess_dataset(42)