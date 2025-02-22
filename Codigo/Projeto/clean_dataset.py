import pandas as pd

# Step 1: Load the dataset
file_path = "C:/Users/tiago.batista/Desktop/mineração/data/data.csv"
df = pd.read_csv(file_path, sep=";", encoding="utf-8")

# Step 2: Inspect the dataset
print(f"Dataset shape: {df.shape}")
print("Column names:")
print(df.columns)
print("Missing values per column:")
print(df.isnull().sum())
print("Target variable distribution:")
print(df['Target'].value_counts())

# Step 3: Handle invalid or missing values
valid_targets = ['Dropout', 'Enrolled', 'Graduate']
df = df[df['Target'].isin(valid_targets)]

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 4: Encode the Target variable
target_map = {'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}
df['Target'] = df['Target'].map(target_map)

# Step 5: Save the cleaned dataset
cleaned_file_path = "cleaned_dataset.csv"
df.to_csv(cleaned_file_path, index=False, sep=";", encoding="utf-8")

print(f"Cleaned dataset saved to: {cleaned_file_path}")