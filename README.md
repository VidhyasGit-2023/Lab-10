# Lab-10
PythonPrograms - BigData Course - Machine Learning

Overview:
The objective of this assignment is to explore advanced Python tools for machine learning. Students will utilize publicly available datasets from the provided GitHub repositories to perform data exploration, and preprocessing, implement machine learning models, and visualize the results using Python programming only.

I have selected the dataset “titanic.csv” from https://github.com/awesomedata/awesome-public-datasets

The Titanic dataset is a classic and widely used dataset in the field of machine learning and data analysis. It contains information about passengers aboard the RMS Titanic, including features like age, gender, ticket class, cabin, fare, and survival status. The dataset is often used for binary classification tasks, with the goal of predicting whether a passenger survived or not based on the given features.
The relevance of the Titanic dataset to machine learning tasks lies in its simplicity and interpretability. It provides a good starting point for beginners in the field of data science to practice and understand basic concepts of data preprocessing, feature engineering, and model building. The dataset is relatively small, which makes it easy to work with and allows for quick experimentation with various machine-learning algorithms. Its clear binary classification objective (survived or not survived) helps in understanding and evaluating the performance of different models effectively.
Additionally, the Titanic dataset can be used for feature importance analysis, as the features have intuitive interpretations (e.g., gender, age, ticket class) that can be analyzed to understand which factors played a significant role in passenger survival. Overall, this dataset serves as a valuable resource for learning and building foundational skills in machine learning and data analysis.

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
