import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel('effectsofdepression.xlsx')

print("=" * 60)
print("RAW DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RENAME COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
COL_MAP = {
    'Gender:': 'Gender',
    'Age:': 'Age',
    'Educational Level': 'Education_Level',
    'Little interest or pleasure in doing things ': 'Dep_Interest',
    'Feeling down, depressed, or hopeless': 'Dep_Hopeless',
    'Trouble falling or staying asleep, or sleeping too much': 'Dep_Sleep',
    'Feeling tired or having little energy': 'Dep_Energy',
    'Poor appetite or overeating': 'Dep_Appetite',
    'Feeling bad about yourself or that you are a failure or not have let yourself or your family down': 'Dep_SelfWorth',
    'Trouble concentrating on things, such as reading the newspaper or watching television': 'Anx_Concentration',
    'Moving or speaking so slowly that other people could have noticed Or being so restless that you have been moving around a lot more than usual': 'Anx_Motor',
    'Thoughts that you would be better off dead or of hurting yourself in some way': 'Anx_SelfHarm',
    'Do you have part-time or full-time job? ': 'Employment',
    'Which of the following best describes your term-time accommodation?': 'Accommodation',
    'How many hours do you spend studying each day?': 'Study_Hours',
    'How many of the electronic gadgets (e.g. mobile phone, computer, laptop, PSP, PS4, Wii, etc.) do you have in your home or your student accommodation/mess/hall?': 'Gadgets_Owned',
    'How many hours do you spend on social media per day?': 'Social_Media_Hours',
    'Your Last Semester GPA: ': 'GPA'
}

df = df.rename(columns=COL_MAP)
print(f"\nColumns after renaming:\n{list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────
df = df.drop_duplicates()
print(f"\nAfter removing duplicates: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FILL MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
df['Gadgets_Owned'] = df['Gadgets_Owned'].fillna(df['Gadgets_Owned'].mode()[0])
print(f"\nMissing values after imputation:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print("(No remaining missing values)" if df.isnull().sum().sum() == 0 else "")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: STANDARDIZE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
df['Employment'] = df['Employment'].str.strip().str.title().replace({'Full Time': 'Full time'})
print(f"\nEmployment unique values: {df['Employment'].unique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CONVERT AND CLEAN GPA
# ─────────────────────────────────────────────────────────────────────────────
df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
df['GPA'] = df['GPA'].clip(1.0, 4.0)
df = df.dropna(subset=['GPA'])
print(f"\nAfter GPA cleaning: {df.shape}")
print(f"GPA range: {df['GPA'].min():.2f} - {df['GPA'].max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
dep_cols = ['Dep_Interest', 'Dep_Hopeless', 'Dep_Sleep', 'Dep_Energy', 'Dep_Appetite', 'Dep_SelfWorth']
anx_cols = ['Anx_Concentration', 'Anx_Motor', 'Anx_SelfHarm']

# Aggregate scores
df['Depression_Score'] = df[dep_cols].sum(axis=1)
df['Anxiety_Score']    = df[anx_cols].sum(axis=1)
df['Total_Score']      = df['Depression_Score'] + df['Anxiety_Score']

# Depression severity (based on PHQ-9 thresholds)
def depression_severity(score):
    if score <= 4:  return 'Minimal'
    elif score <= 9:  return 'Mild'
    elif score <= 14: return 'Moderate'
    else:             return 'Severe'

df['Depression_Severity'] = df['Depression_Score'].apply(depression_severity)

# GPA category
def gpa_category(gpa):
    if gpa >= 3.5:  return 'High (3.5-4.0)'
    elif gpa >= 3.0: return 'Mid-High (3.0-3.49)'
    elif gpa >= 2.5: return 'Mid-Low (2.5-2.99)'
    else:            return 'Low (<2.5)'

df['GPA_Category'] = df['GPA'].apply(gpa_category)

print(f"\nNew columns added: Depression_Score, Anxiety_Score, Total_Score, Depression_Severity, GPA_Category")
print(f"Depression Severity breakdown:\n{df['Depression_Severity'].value_counts()}")
print(f"\nGPA Category breakdown:\n{df['GPA_Category'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE PREPROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
output_path = 'preprocessed_dataset.xlsx'
df.to_excel(output_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSED DATASET")
print("=" * 60)
print(f"Final shape   : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Saved to      : {output_path}")
print("\nSample rows:")
print(df[['Gender', 'Age', 'Education_Level', 'GPA', 'Depression_Score',
          'Anxiety_Score', 'Depression_Severity', 'GPA_Category']].head(5).to_string(index=False))
