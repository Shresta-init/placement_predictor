import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output_log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()
sys.stderr = Logger()

import pandas as pd

# load dataset
df = pd.read_csv("data/student_data.csv")

# show first 5 rows
print(df.head())

# 1. shape of data (rows, columns)
print("Shape of dataset:", df.shape)

# 2. column names
print("\nColumns:")
print(df.columns)

# 3. data info
print("\nInfo:")
print(df.info())

# 4. statistical summary
print("\nSummary:")
print(df.describe())

# 5. missing values
print("\nMissing values:")
print(df.isnull().sum())

# convert Placement_Status to numbers
df['Placement_Status'] = df['Placement_Status'].map({'Placed': 1, 'Not Placed': 0})

print(df['Placement_Status'].head())

# convert categorical columns to numbers
df = pd.get_dummies(df, drop_first=True)

print(df.head())

# define target (output)
y = df['Placement_Status']


# remove leakage columns
df = df.drop(['Placement_Probability', 'Salary_LPA'], axis=1)

# target
y = df['Placement_Status']

# features
X = df.drop(['Placement_Status'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("New Accuracy:", accuracy)

# feature importance (for logistic regression)
importance = model.coef_[0]

feature_names = X.columns

# combine into dataframe
import pandas as pd
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

X = df.drop(['Placement_Status', 'Student_ID', 'College_ID'], axis=1)

# save training columns
import pickle

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

print("Columns saved ")

import pickle

# save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully")

X = df.drop(['Placement_Status'], axis=1)   # DO NOT remove Student_ID now

# after encoding
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)
