import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load the CSV data into a DataFrame
titanic_df = pd.read_csv(csv_file)

# Display the first few rows of the dataset
print(titanic_df.head())

# Summary statistics of the numerical columns
print(titanic_df.describe())

# Check for missing values
print(titanic_df.isnull().sum())

# Data visualization

# Distribution of passenger classes
sns.countplot(x='Pclass', data=titanic_df)
plt.title('Passenger Class Distribution')
plt.show()

# Survival count by gender
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival count by Gender')
plt.show()

# Age distribution
sns.histplot(x='Age', data=titanic_df, kde=True)
plt.title('Age Distribution')
plt.show()

# Survival count based on the port of embarkation
sns.countplot(x='Embarked', hue='Survived', data=titanic_df)
plt.title('Survival count by Embarked Port')
plt.show()