# Lab-10
PythonPrograms - BigData Course - Machine Learning

Overview:
The objective of this assignment is to explore advanced Python tools for machine learning. Students will utilize publicly available datasets from the provided GitHub repositories to perform data exploration, and preprocessing, implement machine learning models, and visualize the results using Python programming only.

Dataset1 - I have selected the dataset “titanic.csv” from https://github.com/awesomedata/awesome-public-datasets

Dataset2 – I have selected the dataset “Ottawa_Public_Library_Locations_2023_” from https://github.com/awesomedata/awesome-public-datasets#government (https://open.ottawa.ca/datasets/a3fffaa13cc94801a7ce4aa244970ffb_0/explore?location=45.315239%2C-75.839900%2C1.76)

Dataset1 - The Titanic dataset is a classic and widely used dataset in the field of machine learning and data analysis. It contains information about passengers aboard the RMS Titanic, including features like age, gender, ticket class, cabin, fare, and survival status. The dataset is often used for binary classification tasks, with the goal of predicting whether a passenger survived or not based on the given features.
The relevance of the Titanic dataset to machine learning tasks lies in its simplicity and interpretability. It provides a good starting point for beginners in the field of data science to practice and understand basic concepts of data preprocessing, feature engineering, and model building. The dataset is relatively small, which makes it easy to work with and allows for quick experimentation with various machine-learning algorithms. Its clear binary classification objective (survived or not survived) helps in understanding and evaluating the performance of different models effectively.
Additionally, the Titanic dataset can be used for feature importance analysis, as the features have intuitive interpretations (e.g., gender, age, ticket class) that can be analyzed to understand which factors played a significant role in passenger survival. Overall, this dataset serves as a valuable resource for learning and building foundational skills in machine learning and data analysis.

Dataset2 - The Ottawa public libraries locations dataset is relevant to machine learning tasks due to its geospatial nature and potential applications in various location-based analyses. This dataset provides information about the locations of public libraries in Ottawa as of 2023. It likely includes attributes such as latitude, longitude, address, branch names, and other related data. Such geospatial datasets are valuable for several machine learning applications.
Geospatial Analysis: The Ottawa public libraries dataset can be used for geospatial analysis, which involves understanding the geographical distribution of public libraries in the city. By employing clustering algorithms, one can identify patterns and group libraries based on their proximity, serving as a basis for optimizing library services and resource allocation.
Recommendation Systems: Machine learning models can leverage this dataset to create location-based recommendation systems. By integrating user preferences and library attributes, personalized recommendations can be made to users based on their geographical location and the characteristics of nearby libraries.
Service Demand Prediction: With historical data on library visits, one could predict future demand for specific libraries, assisting in staff planning and resource allocation.
Accessibility Analysis: The dataset can be used to analyze the accessibility of public libraries in different neighborhoods or regions. This analysis could help policymakers identify underserved areas and improve library access for all residents.
Land Use Planning: City planners can incorporate this dataset into their decision-making processes, ensuring that new developments consider proximity to libraries and promote a strong sense of community.
Marketing and Outreach: Machine learning algorithms can identify potential target audiences for specific library branches, enabling more efficient marketing and outreach campaigns.
In summary, the Ottawa public libraries locations dataset holds significant potential for various machine learning applications, particularly in geospatial analysis, recommendation systems, demand prediction, accessibility analysis, land use planning, and marketing efforts. It is a valuable resource for researchers and policymakers aiming to enhance library services and optimize their distribution throughout the city.


Performed Exploratory Data Analysis EDA for the Titanic dataset and Generated the summary statistics, identify data types, and visualize the data distribution to gain insights into the dataset

Preprocessed the Titanic dataset by handling missing values, and outliers, and perform feature engineering when necessary to prepare the data for machine learning models.

The SVM and Random Forest models use various metrics. The Random Forest model generally outperformed the SVM model in terms of accuracy, precision, recall, and F1-score. The confusion matrix also shows that the Random Forest model has fewer false positives and false negatives compared to SVM. Based on these evaluation metrics, we can conclude that the Random Forest model is more suitable for the Titanic dataset than the SVM model.

Libraries such as Matplotlib and Seaborn create three different visualizations:
Scatter plot: Age vs. Fare
This scatter plot visualizes the relationship between the passengers' ages and the fares they paid. Survived passengers are indicated with different colours (blue for not survived and red for survived).
Heatmap: Correlation matrix
The heatmap shows the correlation between different numerical features of the dataset. It helps to identify patterns and relationships between variables.
Bar chart: Number of survivors by sex
This bar chart displays the number of survivors and non-survivors based on the passengers' sex. The bars are coloured based on survival status (blue for not survived and orange for survived).

Created three additional visualizations using Matplotlib, Seaborn, and Plotly:
Seaborn: Count of Passengers by Pclass and Sex
This count plot displays the number of passengers in each passenger class (Pclass) segregated by sex.
Plotly: Age distribution by Sex and Survived
This histogram shows the age distribution of passengers grouped by sex, with the bars coloured according to their survival status.
Plotly: Fare distribution by Pclass and Embarked
This box plot visualizes the fare distribution across different passenger classes (Pclass) and their embarkation points (Embarked).
