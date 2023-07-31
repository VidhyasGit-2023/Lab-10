import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Assuming you have loaded the dataset into a DataFrame named 'df'
#  Load the dataset and preprocess
df = pd.read_csv('Ottawa_Public_Library_Locations_2023_.csv')

# Example 1: Scatter plot of library locations on a map using Plotly
fig = px.scatter_mapbox(df, lat='F_LATITUDE', lon='F_LONGITUDE', hover_name='Name',
                        color='F_Federal_Electoral_District__3', size_max=15, zoom=10, mapbox_style='carto-positron')
fig.update_layout(title='Geospatial Distribution of Library Branches',
                  margin={'l': 0, 'r': 0, 't': 50, 'b': 0})
fig.show()

# Example 2: Bar chart of the number of libraries in each region using Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='F_Federal_Electoral_District__3', data=df, palette='Set2')
plt.title('Number of Library Branches in Different Regions')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Example 3: Heatmap of the correlation between numeric columns using Seaborn
numeric_columns = ['F_Census_Subdivision_Code', 'F_Federal_Electoral_District__2', 'F_Dissemination_Area_Code']
df_numeric = df[numeric_columns]
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of  District Subdivision areas')
plt.show()

# Example 4: Histogram of a numeric column using Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(df['F_Census_Subdivision_Code'], bins=20, color='skyblue')
plt.title('Histogram of F_Census_Subdivision_Code')
plt.xlabel('F_Census_Subdivision_Code')
plt.ylabel('Frequency')
plt.show()