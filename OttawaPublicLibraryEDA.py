import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract the CSV file
csv_file = "Ottawa_Public_Library_Locations_2023_.csv"

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_file)

# Display basic information about the dataset
print(df.info())

# View the first few rows of the dataset
print(df.head())

# Get summary statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Identify data types of each column
print("\nData Types:")
print(df.dtypes)

# Visualize: Plot the distribution of library branches across the city
plt.figure(figsize=(10, 6))
sns.countplot(x='F_Federal_Electoral_District__3', data=df)
plt.title('Number of Library Branches in Different Regions')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualize: Plot a scatter plot to visualize the latitude and longitude of library branches
plt.figure(figsize=(8, 6))
sns.scatterplot(x='F_LONGITUDE', y='F_LATITUDE', data=df, hue='F_Federal_Electoral_District__3', palette='viridis')
plt.title('Geospatial Distribution of Library Branches')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='best')
plt.show()

