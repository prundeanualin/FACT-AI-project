# Make sure your data is in:
# data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ
# or in:
# data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ

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


def preprocess_dataset():

    print("Start loading original PISA2015 datasets...")

    # Load and process student questionnaire files
    student_df1 = load_dataset(student_questionnaire_file1)
    student_df2 = load_dataset(student_questionnaire_file2)
    # Merge the 2 student questionnaires by adding only non-duplicate columns from qqq_2
    student_df2_new_cols = student_df2[student_df2.columns.difference(student_df1.drop(columns=merge_columns).columns)]

    # Merge the 2 student questionnaire files and keep only the useful columns
    student_df = pd.merge(student_df1, student_df2_new_cols, on=merge_columns)[list_of_useful_columns]

    print("Final student table")
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

    print("\n\nWriting the processed dataframe to csv...")

    final_df.to_csv('data/pisa2015/dataset_not_melted.csv', index=False)
