import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Step 1: Load the dataset and preprocess
df = pd.read_csv('Ottawa_Public_Library_Locations_2023_.csv')

# Keep only numeric columns (you can modify the list based on your dataset)
numeric_columns = ['F_Census_Subdivision_Code', 'F_Federal_Electoral_District__2', 'F_Dissemination_Area_Code']
df_numeric = df[numeric_columns]

# Step 2: Split the data into features (X) and target (y)
X = df_numeric.drop('F_Census_Subdivision_Code', axis=1)  # Replace 'target_column' with the column you want to predict
y = df_numeric['F_Federal_Electoral_District__2']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Step 6: Predict using the SVM model
y_pred_svm = svm_model.predict(X_test_scaled)

# Step 7: Evaluate SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=1))  # Set zero_division=1

# Step 8: Train Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Step 9: Predict using the Random Forest model
y_pred_rf = random_forest_model.predict(X_test)

# Step 10: Evaluate Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=1))  # Set zero_division=1

# Evaluate SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=1)
svm_recall = recall_score(y_test, y_pred_svm, average='weighted', zero_division=1)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=1)
svm_confusion_matrix = confusion_matrix(y_test, y_pred_svm)

# Evaluate Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)
rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=1)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=1)
rf_confusion_matrix = confusion_matrix(y_test, y_pred_rf)

# Print the evaluation metrics
print("SVM Model Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-score:", svm_f1)
print("Confusion Matrix:\n", svm_confusion_matrix)

print("\nRandom Forest Model Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)
print("Confusion Matrix:\n", rf_confusion_matrix)