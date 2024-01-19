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
student_info_columns = ["CNTSCHID", "CNTSTUID", "CNT", "ST004D01T", "Region", "AGE", "HOMEPOS", "HISCED", "OECD"]


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


def reshape_dataset(df, item_columns):
    print("Reshaping dataset...")
    for col in item_columns:
        df[col] = pd.to_numeric(df[col])

    perform_nan_analysis(df, item_columns)

    final_df_reshaped = pd.melt(df, id_vars=student_info_columns, value_vars=item_columns,
                                var_name='item_old_id', value_name='score')
    nr_rows_all = len(final_df_reshaped)
    print(f"\n\nAfter melting (with NaN item scores) - {nr_rows_all}")
    print(final_df_reshaped)

    final_df_reshaped.dropna(subset=['score'], inplace=True)
    nr_rows_not_nan = len(final_df_reshaped)
    print(
        f"\nKeeping only the rows with not NaN scores - {nr_rows_not_nan} - {nr_rows_not_nan / nr_rows_all * 100} % from all the initial rows")
    print(final_df_reshaped)
    return final_df_reshaped


def build_id_vocab(list_of_items):
    vocab = {}
    list_of_items.sort()
    for new_incremental_id, old_id in enumerate(list_of_items):
        vocab[old_id] = new_incremental_id
    return vocab


def postprocess_columns(df, cognitive_items):
    print("Postprocessing columns...")
    df.rename(columns={'CNTSTUID': 'user_id', 'ST004D01T': 'GENDER', 'HISCED': 'EDU', 'HOMEPOS': 'ECONOMIC'},
              inplace=True)
    # Filter for OECD countries to remove rows with 'Not applicable'
    df = df.loc[df['OECD'] < 2]
    df.loc[:, 'OECD'] = df['OECD'] == 1
    # Process the gender to have values 0 and 1
    df.loc[:, 'GENDER'] = df['GENDER'] == 1
    # Remove NaN values and process based on whether the highest parents education is above ISCED 1
    df.dropna(subset=['EDU'], inplace=True)
    df.loc[:, 'EDU'] = df['EDU'] > 1
    df = df.loc[df['score'] < 2]
    # Process based on whether the number of home possessions (wealth index) is above 0
    df.loc[:, 'ECONOMIC'] = df['ECONOMIC'] > 0
    df.loc[:, 'CNT'] = df['CNT'].astype(str).map(lambda x: x.lstrip("b\'").rstrip("\'"))

    # Replace old ids with the new ones from the vocabulary for cognitive items and student id
    df['item_id'] = df['item_old_id']
    cognitive_id_vocab = build_id_vocab(cognitive_items)
    df.loc[:, 'item_id'] = df['item_id'].map(cognitive_id_vocab)
    student_ids = df['user_id'].unique().tolist()
    student_id_vocab = build_id_vocab(student_ids)
    df.loc[:, 'user_id'] = df['user_id'].map(student_id_vocab)
    return df


def preprocess_dataset(random_seed):
    print("Start loading original PISA2015 datasets...")

    # Load and process student questionnaire files
    student_df1 = load_dataset(student_questionnaire_file1)
    student_df2 = load_dataset(student_questionnaire_file2)
    # Merge the 2 student questionnaires by adding only non-duplicate columns from qqq_2
    student_df2_new_cols = student_df2[student_df2.columns.difference(student_df1.drop(columns=merge_columns).columns)]

    # Merge the 2 student questionnaire files and keep only the useful columns
    student_df = pd.merge(student_df1, student_df2_new_cols, on=merge_columns)[student_info_columns]

    print("Final student table with both questionnaires")
    print(student_df.columns)
    print(student_df.head())

    # Load and process cognitive questionnaire file
    cognitive_df = load_dataset(cognitive_file)

    # Build the vocabulary holding the mapping between cognitive item long id and it's name
    cognitive_item_vocab = build_item_vocab()
    # Columns representing the cognitive items(questions) only with a valid score
    cognitive_item_ids = cognitive_item_vocab['item_old_id'].unique().tolist()

    # For the cognitive table, keep only the *merge columns* and the ones that keep the binary *item grade*
    cog_keep_columns = [col for col in merge_columns]
    for c in cognitive_df.columns.tolist():
        if c in cognitive_item_ids:
            cog_keep_columns.append(c)
    cognitive_df = cognitive_df[cog_keep_columns]

    print("Cognitive filtered columns: ")
    print(cognitive_df.columns.tolist())

    # Merge the processed student questionnaire with the cognitive one, based on the merge columns
    final_df = pd.merge(student_df, cognitive_df, on=merge_columns)

    print("\n\nFinal student + cognitive table columns")
    print(final_df.columns.tolist())
    print("Cognitive items:")
    print(cognitive_item_ids)
    print("Final student + cognitive table")
    print(final_df.head())

    final_df = reshape_dataset(final_df, cognitive_item_ids)
    final_df = postprocess_columns(final_df, cognitive_item_ids)
    final_df = pd.merge(final_df, cognitive_item_vocab, on='item_old_id')

    final_df.to_csv('data/pisa2015/dataset.csv', index=False)

    print("Splitting dataset into train/validation/test + attacker train/test sets...")
    print(f"Nr of total items: {len(final_df)}")
    train, validation, test = np.split(final_df.sample(frac=1, random_state=random_seed),
                                       [int(train_split * len(final_df)),
                                        int((train_split + valid_split) * len(final_df))])
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


def trim_df(df, str_to_contain, pos_after_str):
    df.columns.values[0] = 'item_old_id'
    df.columns.values[1] = 'item_name'
    df.columns.values[5] = 'format'
    df.loc[:, 'item_name'] = df['item_name'].astype(str)
    df = df.loc[df['item_name'].str.contains(str_to_contain)]
    df.loc[:, 'item_name'] = df['item_name'].map(lambda x: x.split(str_to_contain)[pos_after_str])
    return df


def build_item_vocab():
    # Read the codebook files
    df = pd.read_excel('data/pisa2015/Codebook_CMB.xlsx', sheet_name='CY6_MS_CMB_STU_COG (Cognitive)')
    df1 = trim_df(df, ' - Q', 0)
    # df2 = pd.read_excel('data/pisa2015/Codebook_CMB.xlsx', sheet_name='CY6_MS_CMB_STU_FLT (Fin. Lit.)')
    # df2 = trim_df(df2, ' - Q', 0)
    # df3 = pd.read_excel('data/pisa2015/Codebook_CMB.xlsx', sheet_name='CY6_MS_CMB_STU_CPS (Pb. Solv.)')
    # df3 = trim_df(df3, ': ', 1)
    # vocab_df = pd.concat([df1, df2, df3]).reset_index(drop=True)
    vocab_df = df1
    vocab_df = vocab_df.loc[vocab_df['format'] == '0 - 1']
    vocab_df = vocab_df[['item_old_id', 'item_name']]
    vocab_df.to_csv('data/pisa2015/item_vocab.csv', index=False)
    return vocab_df

# preprocess_dataset(42)

# df = pd.read_csv('data/pisa2015/dataset.csv')
# items = df['item_id'].unique().tolist()
# students = df['user_id'].unique().tolist()
# print("Number of items: ", len(items))
# print("Number of students: ", len(students))