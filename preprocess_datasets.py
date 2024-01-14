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


def merge_student_questionnaires(df1, df2):
    # Find duplicate columns
    duplicate_columns = df1.columns.intersection(df2.columns)

    # Print duplicate column names
    print("Duplicate column names:", duplicate_columns)

    # Select only non-duplicate columns from qqq_2
    unique_columns_student_df2 = df2.drop(columns=duplicate_columns)

    # Concatenate qqq with the unique columns of qqq_2
    return pd.concat([df1, unique_columns_student_df2], axis=1)

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
# These are the columns on which we will merge the student and the cognitive questionnaires
stud_cog_merge_cols = ["CNT", "CNTSCHID", "CNTSTUID"]

print("Start loading original PISA2015 datasets...")

# Load and process student questionnaire files
student_df1 = load_dataset(student_questionnaire_file1)
student_df2 = load_dataset(student_questionnaire_file2)
# Merge the 2 student questionnaire files and keep only the useful columns
student_df = merge_student_questionnaires(student_df1, student_df2)[list_of_useful_columns]

# Load and process cognitive questionnaire file
cognitive_df = load_dataset(cognitive_file)
# Drop columns that we definitely do not need (they are not merge_columns and neither item column)
cognitive_df.drop(columns=cognitive_df.columns[4:20], inplace=True)
# From those remaining, keep only the *merge columns* and the ones that keep the *item grade*, which in the codebook
# appear either as 'Coded Response' or as 'Scored Response' and end in 'C' or 'S'
cognitive_df = cognitive_df[[x for x in cognitive_df if x in stud_cog_merge_cols or x.endswith(('C', 'S'))]]

# Merge the processed student questionnaire with the cognitive one, based on the merge columns
final_df = pd.merge(student_df, cognitive_df, on=stud_cog_merge_cols)


print(final_df.columns.tolist())
print(final_df.head())
print(final_df.tail())

print("Writing the final processed dataframe to csv...")
final_df.to_csv('data/pisa2015/dataset.csv')

