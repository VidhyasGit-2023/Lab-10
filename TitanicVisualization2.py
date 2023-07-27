import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
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

# Plot: Count of Passengers by Pclass and Sex using Seaborn
plt.figure(figsize=(8, 6))
sns.countplot(x="Pclass", hue="Sex", data=df, palette="coolwarm")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.title("Count of Passengers by Pclass and Sex")
plt.legend(title="Sex", labels=["Male", "Female"])
plt.show()

# Plotly: Age distribution by Sex and Survived using Plotly Express
fig = px.histogram(df, x="Age", color="Survived", facet_col="Sex", nbins=20, title="Age Distribution by Sex and Survived",
                   labels={"Sex": "Sex (0=Male, 1=Female)", "Survived": "Survived (0=No, 1=Yes)"})
fig.update_layout(showlegend=True)
fig.show()

# Plotly: Fare distribution by Pclass and Embarked using Plotly Express
fig = px.box(df, x="Pclass", y="Fare", color="Embarked", points="all", title="Fare Distribution by Pclass and Embarked",
             labels={"Pclass": "Pclass", "Fare": "Fare", "Embarked": "Embarked (0=C, 1=Q, 2=S)"})
fig.update_layout(showlegend=False)
fig.show()