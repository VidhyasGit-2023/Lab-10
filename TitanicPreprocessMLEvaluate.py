import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import zipfile
import io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

# Separate features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

# Evaluate the SVM model
accuracy_svm, precision_svm, recall_svm, f1_svm, cm_svm = evaluate_model(svm_model, X_test, y_test)
print("SVM Evaluation:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-Score:", f1_svm)
print("Confusion Matrix:")
print(cm_svm)

# Evaluate the Random Forest model
accuracy_rf, precision_rf, recall_rf, f1_rf, cm_rf = evaluate_model(rf_model, X_test, y_test)
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-Score:", f1_rf)
print("Confusion Matrix:")
print(cm_rf)