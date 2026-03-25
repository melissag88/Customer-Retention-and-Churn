import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load dataset
file_path = "Telco_customer_churn.xlsx"

df = pd.read_excel(file_path)

print(df.head())

#Data inspection
print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn Label"].value_counts())

#Summary Stats Table
print("\nSummary Statistics:")
print(df.describe())

#Data Cleaning
#Convert to numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

#Check data type and missing values
print(df["Total Charges"].dtype)
print(df["Total Charges"].isnull().sum())

#Drop any rows with missing Total Charges (11 rows)
df = df.dropna(subset=["Total Charges"])

#Checking shape
print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

#Exploratory Data Analysis(EDA)
#Set Seaborn style
sns.set(style="whitegrid")

#1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn Label", hue="Churn Label", palette="Set2", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.show()

#2. Categorical Features vs Churn
categorical_features = [
    "Contract", "Payment Method", "Internet Service",
    "Senior Citizen", "Partner", "Dependents", "Paperless Billing"
]

for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, hue="Churn Label", palette="Set1")
    plt.title(f"{col} vs Customer Churn")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Churn")
    plt.show()

#3. Numeric Features vs Churn
numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="Churn Label", y=col)
    plt.title(f"{col} Distribution by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel(col)
    plt.show()

#4. Correlation Heatmap
numeric_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "Churn Value", "CLTV"]

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#5. Churn Rate Heatmap
#Create churn rate table
churn_rate = pd.crosstab(
    df["Contract"],
    df["Internet Service"],
    values=df["Churn Value"],
    aggfunc="mean"
)

#Plot heatmap
plt.figure(figsize=(8,5))
sns.heatmap(churn_rate, annot=True, fmt=".2f", cmap="Reds")

plt.title("Churn Rate by Contract Type and Internet Service")
plt.xlabel("Internet Service Type")
plt.ylabel("Contract Type")

plt.show()

#6. Churn Rate by Customer Tenure
#Create tenure groups (bins)
df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"]
)

#Calculate churn rate for each tenure group
tenure_churn = df.groupby("Tenure Group", observed=False)["Churn Value"].mean()

#Plot
plt.figure(figsize=(8,5))
sns.barplot(x=tenure_churn.index, y=tenure_churn.values)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Churn Rate")

plt.show()

#Prepping data for ML

#Remove unnecessary columns
columns_to_drop = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Count",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "Tenure Group"
]

df = df.drop(columns=columns_to_drop)

print(df.head())

#Summary Stats Table
print("\nSummary Statistics:")
print(df.describe())

print("\nColumn names:")
print(df.columns)

# Export cleaned dataset for Tableau
df.to_csv("telco_churn_cleaned.csv", index=False)

#Define Target(y) and Features(x)

# Target variable
y = df["Churn Value"]

# Features
X = df.drop(columns=["Churn Label", "Churn Value"])


#Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

#Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Check dataset size
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nColumn names:")
print(df.columns)

#Import scaler
from sklearn.preprocessing import StandardScaler

#Scale training and test data
#Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Train Model

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

#Evaluate the model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Feature Importance (Logistic Regression)

#Get feature names
feature_names = X.columns

#Get coefficients
coefficients = model.coef_[0]

#Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

#Sort by importance (absolute value)
feature_importance["Abs Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(by="Abs Coefficient", ascending=False)

#Show top 10
top_features = feature_importance.head(10)

print(top_features)

#Visualize top churn drivers
#Plot top 10 features
#Sort so positive and negative are separated visually
top_features = top_features.sort_values(by="Coefficient")

plt.figure(figsize=(10,6))
sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature"
)

plt.axvline(0)  #Adds vertical line at 0

plt.title("Top 10 Factors Influencing Customer Churn")
plt.xlabel("Impact on Churn")
plt.ylabel("Feature")

plt.show()

#Train Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

#Evaluate the model

# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


#Feature Importance (Random Forest)

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(by="Importance", ascending=False)


# Top 10 features
rf_top_features = rf_importance.head(10)

print(rf_top_features)


#Visualize Top Features

plt.figure(figsize=(10,6))
sns.barplot(
    data=rf_top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.show()