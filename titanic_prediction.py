# Titanic Survival Prediction
# CodSoft Internship - Task 1
# Author: Aarthi S

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Load the Dataset
# -----------------------------
data = pd.read_csv("train.csv")

print("First 5 rows of the dataset:\n")
print(data.head())

# -----------------------------
# Step 2: Select Required Columns
# -----------------------------
selected_data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
selected_data['Age'] = selected_data['Age'].fillna(
    selected_data['Age'].mean()
)

# -----------------------------
# Step 4: Convert Categorical Data
# -----------------------------
gender_encoder = LabelEncoder()
selected_data['Sex'] = gender_encoder.fit_transform(
    selected_data['Sex']
)

# -----------------------------
# Step 5: Split Features & Target
# -----------------------------
X = selected_data.drop('Survived', axis=1)
y = selected_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# Step 6: Train the Model
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Model Evaluation
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# Step 8: Survival Details from Dataset
# -----------------------------
print("\nPassenger Survival Details (Sample):\n")

sample_data = data[['PassengerId', 'Sex', 'Age', 'Pclass', 'Survived']].head(10)

for index, row in sample_data.iterrows():
    status = "Survived" if row['Survived'] == 1 else "Did NOT Survive"
    print(
        f"Passenger ID {row['PassengerId']} | "
        f"Gender: {row['Sex']} | "
        f"Age: {row['Age']} | "
        f"Class: {row['Pclass']} | "
        f"Status: {status}"
    )

# -----------------------------
# Step 9: Predict for a New Passenger
# -----------------------------
# Format: [Pclass, Sex (0=Female,1=Male), Age, Fare]
new_passenger = [[2, 1, 28, 20.5]]
result = model.predict(new_passenger)

print("\nNew Passenger Prediction:")
if result[0] == 1:
    print("The passenger is likely to SURVIVE.")
else:
    print("The passenger is likely to NOT survive.")