# Make sure your initial PISA data is in:
# data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ
# or in:
# data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ

import numpy as np
import pandas as pd
import os
import pickle

# Pick SPSS or SAS
file_type = 'SAS'
formats = {'SAS': '.sas7bdat', 'SPSS': '.sav'}

base_path = '../data/pisa2015/'
student_directory = f"PUF_{file_type}_COMBINED_CMB_STU_QQQ/"
cognitive_directory = f"PUF_{file_type}_COMBINED_CMB_STU_COG/"

student_questionnaire_file1 = student_directory + 'cy6_ms_cmb_stu_qqq'
student_questionnaire_file2 = student_directory + 'cy6_ms_cmb_stu_qq2'
cognitive_file = cognitive_directory + 'cy6_ms_cmb_stu_cog'
codebook_file, codebook_cog_sheet_name = 'Codebook_CMB.xlsx', 'CY6_MS_CMB_STU_COG (Cognitive)'

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

    # Check how many missing responses are there for each item
    perform_nan_analysis(df, item_columns)

    # Reshape the table so that instead of having a row per user and a column for each item, we have
    # multiple rows per user with all the item columns aggregated into 2 columns: [item_old_id, score]
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


def build_id_remapping(list_of_items):
    # Remap the old id to a new incremental one
    return {old_id: new_id for new_id, old_id in enumerate(sorted(list_of_items))}


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
    df.loc[:, 'item_id'] = df['item_id'].map(build_id_remapping(cognitive_items))
    student_ids = df['user_id'].unique().tolist()
    df['user_old_id'] = df['user_id']
    df.loc[:, 'user_id'] = df['user_id'].map(build_id_remapping(student_ids))
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

    # For the cognitive table, keep only the *merge columns* and the ones that represent a valid question
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
    print("Nr of cognitive items:")
    print(len(cognitive_item_ids))
    print("Final student + cognitive table")
    print(final_df.head())

    final_df = reshape_dataset(final_df, cognitive_item_ids)
    final_df = postprocess_columns(final_df, cognitive_item_ids)
    final_df = pd.merge(final_df, cognitive_item_vocab, on='item_old_id')
    build_item2knowledge(final_df, cognitive_item_ids)

    df_grouped_by_user = final_df.groupby(['user_id'])['item_id'].count()
    print("Items per user analysis:")
    print("Average number of items per user: ", df_grouped_by_user.mean())
    print("Maximum number of items filled in by a user: ", df_grouped_by_user.max())
    print("Minimum number of items filled in by a user: ", df_grouped_by_user.min())

    final_df.to_csv(f'{base_path}dataset.csv', index=False)

    print("Splitting dataset into train/validation/test...")
    print(f"Nr of total items: {len(final_df)}")
    train, validation, test = np.split(final_df.sample(frac=1, random_state=random_seed),
                                       [int(train_split * len(final_df)),
                                        int((train_split + valid_split) * len(final_df))])
    print("Length of train set: " + str(len(train)))
    print("Length of valid set: " + str(len(validation)))
    print("Length of test set: " + str(len(test)))

    print("Splitting dataset into attacker train/test...")
    # The attacker only needs one entry per user
    attacker_dataset = final_df.drop_duplicates(subset=['user_id'])
    remaining_perc = len(attacker_dataset) / len(final_df) * 100
    remaining_perc = "{:.2f}".format(remaining_perc)
    print(f"Attacker dataset size: {len(attacker_dataset)} - {remaining_perc} % of the model training size")
    attacker_train, attacker_test = np.split(attacker_dataset.sample(frac=1, random_state=random_seed),
                                             [int(attacker_train_split * len(attacker_dataset))])

    print("\n\nWriting the final processed train/validate/test + attacker train/test dataframes to csv...")

    train.to_csv(f'{base_path}pisa.train.csv', index=False)
    validation.to_csv(f'{base_path}pisa.validation.csv', index=False)
    test.to_csv(f'{base_path}pisa.test.csv', index=False)

    attacker_train.to_csv(f'{base_path}pisa.attacker.train.csv', index=False)
    attacker_test.to_csv(f'{base_path}pisa.attacker.test.csv', index=False)
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
    df = pd.read_excel(base_path + codebook_file, sheet_name=codebook_cog_sheet_name)
    df1 = trim_df(df, ' - Q', 0)
    # df2 = pd.read_excel(f'{base_path}Codebook_CMB.xlsx', sheet_name='CY6_MS_CMB_STU_FLT (Fin. Lit.)')
    # df2 = trim_df(df2, ' - Q', 0)
    # df3 = pd.read_excel(f'{base_path}Codebook_CMB.xlsx', sheet_name='CY6_MS_CMB_STU_CPS (Pb. Solv.)')
    # df3 = trim_df(df3, ': ', 1)
    # vocab_df = pd.concat([df1, df2, df3]).reset_index(drop=True)
    vocab_df = df1
    vocab_df = vocab_df.loc[vocab_df['format'] == '0 - 1']
    vocab_df = vocab_df[['item_old_id', 'item_name']]
    return vocab_df


def build_item2knowledge(final_df, cognitive_items):
    print("Creating the item2knowledge matrix...")
    df_only_unique_items = final_df.drop_duplicates(subset=['item_old_id'])
    knowledge_domains = sorted(df_only_unique_items['item_name'].unique().tolist())
    print("Nr of knowledge domains: ", len(knowledge_domains))
    knowledge_id_mapping = {knowledge: i for i, knowledge in enumerate(knowledge_domains)}
    item2knowledge = np.zeros((len(cognitive_items), len(knowledge_domains)))
    for idx, row in df_only_unique_items.iterrows():
        c_id = row['item_id']
        k_id = knowledge_id_mapping[row['item_name']]
        item2knowledge[c_id][k_id] = 1.0
    with open(f'{base_path}item2knowledge.pkl', 'wb') as handle:
        pickle.dump(item2knowledge, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return item2knowledge


def split_df(df, seed, ratios):
    """
    Split the dataset based on the user id's, so that there is no common user in the resulting splits.
    :param ratios: list of ratios to split the dataset
    """
    # Reshuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Get the unique use ids
    user_ids = df['user_id'].unique().tolist()
    nr_users = len(user_ids)
    # Split based on the user ids
    splits = []
    ratio_idx = 0
    for r in ratios:
        prev_ratio_idx = ratio_idx
        ratio_idx += r
        split = df[df['user_id'].isin(user_ids[int(prev_ratio_idx * nr_users):int(ratio_idx * nr_users)])]
        splits.append(split)
    final_split = df[df['user_id'].isin(user_ids[int(ratio_idx * nr_users):])]
    splits.append(final_split)
    return splits

# preprocess_dataset(42)

# df_train = pd.read_csv('data/pisa2015/pisa.train.csv')
# df_attacker_train = pd.read_csv('data/pisa2015/pisa.attacker.train.csv')
#
# print("Ratio attacker from all is: ", len(df_attacker_train)/len(df_train))

# items = df[]
# print(len(df[]))
# preprocess_dataset(42)
# df = pd.read_csv('data/pisa2015/dataset.csv')
# items = df['item_id'].unique().tolist()
# students = df['user_id'].unique().tolist()
# print("Number of items: ", len(items))
# print("Number of students: ", len(students))