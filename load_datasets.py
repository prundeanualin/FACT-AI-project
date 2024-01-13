# Make sure your data is in:
# data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ
# or in:
# data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ

from sas7bdat import SAS7BDAT
import pyreadstat
import pandas as pd

# Pick SPSS or SAS
file_type = 'SPSS'

path = 'data/pisa2015/'
qqq_name = 'cy6_ms_cmb_stu_qqq'
qqq2_name = 'cy6_ms_cmb_stu_qq2'

if file_type == "SPSS":
    qqq, meta = pyreadstat.read_sav(path + "PUF_SPSS_COMBINED_CMB_STU_QQQ/" + qqq_name + ".sav")
    qqq_2, meta_2 = pyreadstat.read_sav(path + "PUF_SPSS_COMBINED_CMB_STU_QQQ/" + qqq2_name + ".sav")
elif file_type == "SAS":
    with SAS7BDAT(path + "PUF_SAS_COMBINED_CMB_STU_QQQ/" + qqq_name + ".sas7bdat") as file:
        qqq = file.to_data_frame()
    with SAS7BDAT(path + "PUF_SAS_COMBINED_CMB_STU_QQQ/" + qqq2_name + ".sas7bdat") as file:
        qqq_2 = file.to_data_frame()
else:
    raise Exception("Incorrect file type, pick SAS or SPSS.")

# Find duplicate columns
duplicate_columns = qqq.columns.intersection(qqq_2.columns)

# Print duplicate column names
print("Duplicate column names:", duplicate_columns)

# Select only non-duplicate columns from qqq_2
unique_columns_qqq_2 = qqq_2.drop(columns=duplicate_columns)

# Concatenate qqq with the unique columns of qqq_2
df = pd.concat([qqq, unique_columns_qqq_2], axis=1)

print(df.columns.tolist())
print(df.head())
print(df.tail())

# Student International Grade (Derived) - perhaps this is the grade received idk
# variables = {"ST004D01T": "Gender", "Region": "Region", "ST005Q01TA": "Edu_mother_highest",
#              "ST006Q01TA": "Edu_mother_lvl6", "ST006Q02TA": "Edu_mother_lvl5A",
#              "ST006Q03TA": "Edu_mother_lvl5B", "ST006Q04TA": "Edu_mother_lvl4", "ST007Q01TA": "Edu_father_highest",
#              "ST008Q01TA": "Edu_father_lvl6", "ST008Q02TA": "Edu_father_lvl5A", "ST008Q03TA": "Edu_father_lvl5B",
#              "ST008Q04TA": "Edu_father_lvl4", "AGE": "Age", "HOMEPOS": "Home_possesions",
#              "PARED": "Index_highest_parental_education", "HISCED": "Highest_Education_of_parents_(ISCED)"}

# The variables we actually need are below. I included all variables I thought we might need in variables above,
# but for example HISCED
# is an aggregate of all edu_mother and edu_father variables so we can just use HISCED
# also we want to include identifiers
variables_using = {"CNTSCHID": "School_id", "CNTSTUID": "Student_id", "CNT": "Country_code",
                   "ST004D01T": "Gender", "Region": "Region", "AGE": "Age", "HOMEPOS": "Home_possesions",
                   "HISCED": "Highest_Education_of_parents_(ISCED)", "OECD": "Whether_OECD_country"}

for var, name in variables_using.items():
    print(name)
    unique_values = df[var].unique()  # Getting unique values
    print(unique_values)
    print()

