import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("loan approval data.csv")

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns


num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


# # Exploratory Data Analysis


# classes_count = df["Loan_Approved"].value_counts()
# plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%")
# plt.title("Is Loan approved or not")

# gender_count = df["Gender"].value_counts()
# ax = sns.barplot(gender_count)
# ax.bar_label(ax.containers[0])

# edu_count = df["Education_Level"].value_counts()
# ax = sns.barplot(edu_count)
# ax.bar_label(ax.containers[0])

# sns.histplot(
#     data=df,
#     x="Applicant_Income",
#     bins=20
# )

# fig, axes = plt.subplots(2, 2)
# sns.boxplot(ax=axes[0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
# sns.boxplot(ax=axes[0, 1], data=df, x="Loan_Approved", y="Credit_Score")
# sns.boxplot(ax=axes[1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
# sns.boxplot(ax=axes[1, 1], data=df, x="Loan_Approved", y="Savings")

# sns.histplot(
#     data=df,
#     x="Credit_Score",
#     hue="Loan_Approved",
#     bins=20,
#     multiple="dodge"
# )
# plt.show()


# # Encoding


le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])


cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)



# # Correlation Heatmap


num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()

plt.figure(figsize=(15, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)


num_cols.corr()["Loan_Approved"].sort_values(ascending=False)


# # Train-Test-Split + Feature Scaling


X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_scaled


# # Train &  Evaluate Models


nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# -------------------------------------------------------------------Evaluation----------------------------------------------------------------------
# print("Naive Bayes model")
# print("Precision: ", precision_score(y_test, y_pred))
# print("Recalll: ", recall_score(y_test, y_pred))
# print("F1 score: ", f1_score(y_test, y_pred))
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("CM: ", confusion_matrix(y_test, y_pred))





