import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Extract the CSV file
csv_file = "Ottawa_Public_Library_Locations_2023_.csv"

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_file)

# Drop rows with any missing values
df.dropna(inplace=True)

# Fill missing values in numerical columns with the mean
df['F_Federal_Electoral_District__2'].fillna(df['F_Federal_Electoral_District__2'].mean(), inplace=True)

# Fill missing values in categorical columns with the most frequent value
df['F_Federal_Electoral_District__2'].fillna(df['F_Federal_Electoral_District__2'].mode()[0], inplace=True)

# Identify outliers in a numerical column using IQR
Q1 = df['F_Aggregate_Dissemination_Area_'].quantile(0.25)
Q3 = df['F_Aggregate_Dissemination_Area_'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['F_Aggregate_Dissemination_Area_'] >= lower_bound) & (df['F_Aggregate_Dissemination_Area_'] <= upper_bound)]

# Alternatively, you can cap the outliers to a specific value
df['F_Aggregate_Dissemination_Area_'] = df['F_Aggregate_Dissemination_Area_'].clip(lower_bound, upper_bound)

X = df.drop('F_Aggregate_Dissemination_Area1', axis=1)  # Replace 'target_column' with the column you want to predict
y = df['F_Aggregate_Dissemination_Area1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Final processed dataset
print(X_test,y_test)
print(df.head())