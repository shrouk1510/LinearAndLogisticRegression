
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# a) Load the "loan_old.csv" dataset.
loan_old = pd.read_csv("loan_old.csv")



# b) Analysis on the dataset
# i) Check for missing values
missing_values = loan_old.isnull().sum()
print("Missing Values:")
print(missing_values)


# ii) Check the type of each feature
feature_types = loan_old.drop(['Max_Loan_Amount','Loan_Status'] ,axis=1).dtypes
print("\nFeature Types:")
print(feature_types)


# iii) Check whether numerical features have the same scale
numerical_features = loan_old.select_dtypes(include=[np.number]).drop('Max_Loan_Amount',axis=1).columns
feature_scales = loan_old[numerical_features].max() - loan_old[numerical_features].min()
print("\nFeatures Scale:")
print(feature_scales)

# Visualize a pairplot between numerical columns
sns.pairplot(loan_old[numerical_features])
plt.show()



# c) Preprocess the data
# i) Remove records containing missing values
loan_old = loan_old.dropna()

# ii) Separate features and targets
X = loan_old.drop(['Max_Loan_Amount', 'Loan_Status'], axis=1)
y_max_loan_amount = loan_old['Max_Loan_Amount']
y_loan_acceptance_status = loan_old['Loan_Status']


# iii) Shuffle and split into training and testing sets
X_train, X_test, y_max_loan_amount_train, y_max_loan_amount_test, y_loan_acceptance_status_train, y_loan_acceptance_status_test = train_test_split(
    X, y_max_loan_amount, y_loan_acceptance_status, test_size=0.2, random_state=132)



# iv) Categorical features encoding
# Identify categorical features
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Property_Area']
# (One-Hot Encoding)
encoder = OneHotEncoder()
loan_old_encoded = pd.get_dummies(loan_old, columns=categorical_features, drop_first=True)

# Display the encoded dataset
print("\nEncoded features in loan old : \n", loan_old_encoded)

# v) categorical targets are encoded
# Identify categorical target columns
categorical_targets = ['Loan_Status']
# Create a LabelEncoder instance for each categorical target column
label_encoder = LabelEncoder()
for column in categorical_targets:
    loan_old[column] = label_encoder.fit_transform(loan_old[column])

# Display the encoded dataset
print("\nEncoded targets in loan old :")
print(loan_old)

# Load the second dataset
loan_new = pd.read_csv("loan_new.csv")

loan_new = loan_new.dropna()



# iv) Categorical features encoding (One-Hot Encoding)
# Identify categorical features
encoder = OneHotEncoder()
loan_new_encoded = pd.get_dummies(loan_new, columns=categorical_features, drop_first=True)

# Display the encoded dataset
print("\nEncoded features in loan new :")
print(loan_new_encoded)

# vi) numerical features are standardized
# Identify numerical features
numerical_features = ['Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History']

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training set and transform the training set
loan_old[numerical_features] = scaler.fit_transform(loan_old[numerical_features])

# Transform the test set using the scaler fitted on the training set
loan_old[numerical_features] = scaler.transform(loan_old[numerical_features])
# Fit the scaler on the training set and transform the training set
loan_new[numerical_features] = scaler.fit_transform(loan_new[numerical_features])

# Transform the test set using the scaler fitted on the training set
loan_new[numerical_features] = scaler.transform(loan_new[numerical_features])

# Display the standardized datasets
print("\nStandardized numerical features in loan old:")
print(loan_old)

print("\nStandardized numerical features in loan new:")
print(loan_new)


# Fit linear regression model to training data

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_loan_acceptance_status_train)

model = LinearRegression()

column_to_drop = 'Loan_ID'
X_train_new = X_train.drop(column_to_drop, axis=1)


# Assuming 'X_train_new' has multiple categorical columns
categorical_columns = ['Gender', 'Education', 'Married', 'Dependents', 'Property_Area']

# One-hot encode each categorical column
for column in categorical_columns:
    encoded_columns = pd.get_dummies(X_train_new[column], prefix=column, drop_first=True)
    X_train_new = pd.concat([X_train_new, encoded_columns], axis=1)
    X_train_new.drop(column, axis=1, inplace=True)

# Now, 'X_train_new' contains one-hot encoded values for all categorical columns
# Continue with fitting your model using 'X_train_new'
#print(X_train_new)


model.fit(X_train_new, y_train_encoded)


# Assess fit model on the test data
X_test_new = X_test.drop(column_to_drop, axis=1)
y_test_encoded = label_encoder.transform(y_loan_acceptance_status_test)

# Assuming 'X_train_new' has multiple categorical columns
categorical_columns2 = ['Gender', 'Education', 'Married', 'Dependents', 'Property_Area']

# One-hot encode each categorical column
for c in categorical_columns2:
    encoded_columns2 = pd.get_dummies(X_test_new[c], prefix=c, drop_first=True)
    X_test_new = pd.concat([X_test_new, encoded_columns2], axis=1)
    X_test_new.drop(c, axis=1, inplace=True)


prediction = model.predict(X_test_new)

print(prediction)

R2Score = r2_score(y_test_encoded, prediction)
print(f'R2 Score = {R2Score}')


X_new = loan_new

X_new = X_new.drop(column_to_drop, axis=1)


for c in categorical_columns2:
    encoded_columns_new = pd.get_dummies(X_new[c], prefix=c, drop_first=True)
    X_new = pd.concat([X_new, encoded_columns_new], axis=1)
    X_new.drop(c, axis=1, inplace=True)


prediction2 = model.predict(X_new)

print(prediction2)


# f) Fit a logistic regression model to the data to predict the loan status.
# Implement logistic regression from scratch using gradient descent.

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
    m = X.shape[0]
    n = X.shape[1]
    # Initialize weights and bias
    w = np.zeros(n)
    b = 0
    for i in range(num_iterations):
        # Calculate predicted probabilities
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        # Calculate gradients for weights and bias
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        # Update weights and bias
        w -= learning_rate * dw.astype('float64')  # Convert dw to float64
        b -= learning_rate * db
    return w, b

# Fit logistic regression model to training data
X_train_logistic = X_train_new.loc[:, X_train_new.columns != 'Max_Loan_Amount']
y_train_logistic = (y_loan_acceptance_status_train == 'Y').astype(int)

learning_rate = 0.01
num_iterations = 1000
w, b = logistic_regression(X_train_logistic.values.astype('float64'), y_train_logistic.values.astype('float64'), learning_rate, num_iterations)

# g) Write a function (from scratch) to calculate the accuracy of the model.

def pred(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred > 0.5).astype(int)


def calculate_accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

# Apply logistic regression model to test data and calculate accuracy
X_test_logistic = X_test_new.loc[:, X_test_new.columns != 'Max_Loan_Amount']
y_test_logistic = (y_loan_acceptance_status_test == 'Y').astype(int)

y_pred_logistic = pred(X_test_logistic.values.astype('float64'), w, b)
accuracy_logistic = calculate_accuracy(y_test_logistic.values, y_pred_logistic)

print(f"Logistic Regression Accuracy: {accuracy_logistic}")

# Make predictions for loan status

X_new_loan_status = X_new.loc[:, X_new.columns != 'Max_Loan_Amount']

# Predict loan acceptance status using logistic regression model
predictions_logistic = pred(X_new_loan_status.values.astype('float64'), w, b)

# Convert binary predictions to 'Y' or 'N'
predictions_logistic = np.where(predictions_logistic == 1, 'Y', 'N')


print("\nPredictions for Loan Status:")
print(predictions_logistic)