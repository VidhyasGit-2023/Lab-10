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

# Load the Titanic dataset (you can use the code provided in the previous answer to load the data)

# Handling missing values
df = pd.read_csv(csv_file)


# Drop irrelevant columns and handle missing values
df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Encode categorical features
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Scatter plot: Age vs. Fare
plt.figure(figsize=(8, 6))
plt.scatter(df["Age"], df["Fare"], c=df["Survived"], cmap="coolwarm", alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Scatter plot: Age vs. Fare (Survived=1, Not Survived=0)")
plt.colorbar(label="Survived")
plt.show()

# Heatmap: Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Bar chart: Number of survivors by sex
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="coolwarm")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Number of Survivors by Sex (Survived=1, Not Survived=0)")
plt.legend(title="Survived", labels=["Not Survived", "Survived"])
plt.show()