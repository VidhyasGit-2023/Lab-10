import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import zipfile
import io

# Define the URL of the ZIP file
zip_url = "https://github.com/awesomedata/awesome-public-datasets/raw/master/Datasets/titanic.csv.zip"

# Download the ZIP file
response = requests.get(zip_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the CSV file from the ZIP archive
csv_file = zip_file.open('titanic.csv')

# Load the Titanic dataset (you can use the code provided in the previous answer to load the data)

# Handling missing values
titanic_df = pd.read_csv(csv_file)

# Check for missing values in each column
print(titanic_df.isnull().sum())

# Replace missing 'Age' values with the median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Replace missing 'Embarked' values with the most common port of embarkation
most_common_embarked = titanic_df['Embarked'].mode()[0]
titanic_df['Embarked'].fillna(most_common_embarked, inplace=True)

# Drop the 'Cabin' column as it has too many missing values
titanic_df.drop('Cabin', axis=1, inplace=True)

# Encoding categorical variables

# Encode 'Sex' column (male: 0, female: 1)
label_encoder = LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])

# Encode 'Embarked' column using one-hot encoding
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True)

# Scaling numerical features

# Standardize 'Age' and 'Fare' columns using StandardScaler
scaler = StandardScaler()
titanic_df[['Age', 'Fare']] = scaler.fit_transform(titanic_df[['Age', 'Fare']])

# Final processed dataset
print(titanic_df.head())