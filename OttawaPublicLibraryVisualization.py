import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have loaded the dataset into a DataFrame named 'df'
#  Load the dataset and preprocess
df = pd.read_csv('Ottawa_Public_Library_Locations_2023_.csv')

# Example 1: Scatter plot of library locations on a map
plt.figure(figsize=(10, 8))
sns.scatterplot(x='F_LONGITUDE', y='F_LATITUDE', data=df, hue='Name', palette='viridis')
plt.title('Geospatial Distribution of Library Branches')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='best')
plt.show()

# Example 2: Bar chart of the number of libraries in each region
plt.figure(figsize=(10, 6))
sns.countplot(x='Name', data=df, palette='Set2')
plt.title('Number of Library Branches in Different Regions')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Example 3: Heatmap of the correlation between numeric columns
numeric_columns = ['F_Census_Subdivision_Code', 'F_Federal_Electoral_District__2', 'F_Dissemination_Area_Code']
df_numeric = df[numeric_columns]
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of District Subdivision areas')
plt.show()

# Example 4: Histogram of a numeric column
plt.figure(figsize=(8, 6))
sns.histplot(df['F_Census_Subdivision_Code'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of F_Census_Subdivision_Code')
plt.xlabel('F_Census_Subdivision_Code')
plt.ylabel('Frequency')
plt.show()