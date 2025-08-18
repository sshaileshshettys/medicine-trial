# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # -------------------------
# # Step 1: Load & Prepare Data
# # -------------------------
# df_large = pd.read_csv('NCT04414150.csv')

# X = df_large[['Age', 'Previous Trials', 'Product Experience']]
# y = df_large['Last Trial Outcome']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # -------------------------
# # Step 2: Train Linear Regression
# # -------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)

# # -------------------------
# # Step 3: Predictions
# # -------------------------
# y_pred_cont = model.predict(X_test)              # continuous predictions
# y_pred = np.where(y_pred_cont >= 0.5, 1, 0)      # convert to 0/1

# # -------------------------
# # Step 4: Evaluation
# # -------------------------
# print("Model Evaluation:")
# print("=" * 50)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print(f"\nMSE: {mean_squared_error(y_test, y_pred_cont):.4f}")
# print(f"R^2 Score: {r2_score(y_test, y_pred_cont):.4f}")

# # -------------------------
# # Step 5: Coefficients (Feature Importance)
# # -------------------------
# coef_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_
# }).sort_values(by='Coefficient', ascending=False)

# plt.figure(figsize=(8, 5))
# sns.barplot(x='Coefficient', y='Feature', data=coef_df)
# plt.title('Linear Regression Coefficients')
# plt.tight_layout()
# plt.savefig('linear_regression_coefficients.png', dpi=300)
# plt.show()

# # -------------------------
# # Step 6: Save Model
# # -------------------------
# joblib.dump(model, 'trial_outcome_regression.pkl')
# print("\nModel saved as 'trial_outcome_regression.pkl'")

# # -------------------------
# # Step 7: Prediction Report
# # -------------------------
# def generate_prediction_report(model, X_test, y_test, y_pred_cont, y_pred):
#     results = []
#     for i in range(len(X_test)):
#         instance = X_test.iloc[i]
#         actual = y_test.iloc[i]
#         pred_cont = y_pred_cont[i]
#         pred_binary = y_pred[i]
#         results.append({
#             'instance': i+1,
#             'Age': instance['Age'],
#             'Previous Trials': instance['Previous Trials'],
#             'Product Experience': instance['Product Experience'],
#             'actual_outcome': 'Positive' if actual == 1 else 'Negative',
#             'predicted_outcome': 'Positive' if pred_binary == 1 else 'Negative',
#             'predicted_score': round(pred_cont, 3),
#             'correct': 'Yes' if actual == pred_binary else 'No'
#         })
#     return pd.DataFrame(results)

# report_df = generate_prediction_report(model, X_test, y_test, y_pred_cont, y_pred)
# print("\nPrediction Report (first 20 rows):")
# print("=" * 50)
# print(report_df.head(20))


# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # -------------------------
# # Step 1: Load & Prepare Data
# # -------------------------
# df_large = pd.read_csv('NCT04414150.csv')

# X = df_large[['Age', 'Previous Trials', 'Product Experience']]
# y = df_large['Last Trial Outcome']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # -------------------------
# # Step 2: Train Logistic Regression
# # -------------------------
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)

# # -------------------------
# # Step 3: Predictions
# # -------------------------
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1 (Positive)

# # -------------------------
# # Step 4: Evaluation
# # -------------------------
# print("Model Evaluation:")
# print("=" * 50)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=['Pred Negative', 'Pred Positive'],
#             yticklabels=['Actual Negative', 'Actual Positive'])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig("logistic_confusion_matrix.png", dpi=300)
# plt.show()

# # -------------------------
# # Step 5: Feature Importance (Coefficients)
# # -------------------------
# coef_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_[0]
# }).sort_values(by='Coefficient', ascending=False)

# plt.figure(figsize=(8, 5))
# sns.barplot(x='Coefficient', y='Feature', data=coef_df)
# plt.title('Logistic Regression Coefficients (Feature Importance)')
# plt.tight_layout()
# plt.savefig('logistic_regression_coefficients.png', dpi=300)
# plt.show()

# # -------------------------
# # Step 6: Save Model
# # -------------------------
# joblib.dump(model, 'trial_outcome_logistic.pkl')
# print("\nModel saved as 'trial_outcome_logistic.pkl'")

# # -------------------------
# # Step 7: Prediction Report
# # -------------------------
# def generate_prediction_report(model, X_test, y_test, y_pred, y_prob):
#     results = []
#     for i in range(len(X_test)):
#         instance = X_test.iloc[i]
#         actual = y_test.iloc[i]
#         pred_binary = y_pred[i]
#         prob = y_prob[i]
#         results.append({
#             'instance': i+1,
#             'Age': instance['Age'],
#             'Previous Trials': instance['Previous Trials'],
#             'Product Experience': instance['Product Experience'],
#             'actual_outcome': 'Positive' if actual == 1 else 'Negative',
#             'predicted_outcome': 'Positive' if pred_binary == 1 else 'Negative',
#             'predicted_probability': round(prob, 3),
#             'correct': 'Yes' if actual == pred_binary else 'No'
#         })
#     return pd.DataFrame(results)

# report_df = generate_prediction_report(model, X_test, y_test, y_pred, y_prob)
# print("\nPrediction Report (first 20 rows):")
# print("=" * 50)
# print(report_df.head(20))



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # -----------------------
# # Generate dataset
# # -----------------------
# num_new_entries = 10000
# user_ids = [f'NTC{str(i).zfill(4)}' for i in range(4, num_new_entries + 4)]

# data = {
#     'User ID': user_ids,
#     'Age': np.random.randint(18, 80, size=num_new_entries),
#     'Previous Trials': np.random.randint(0, 6, size=num_new_entries),
#     'Last Trial Outcome': np.random.choice([0, 1], size=num_new_entries),
#     'Product Experience': np.random.choice([0, 1], size=num_new_entries),
# }

# df = pd.DataFrame(data)

# # -----------------------
# # Features (X) and Target (y)
# # -----------------------
# X = df[['Age', 'Previous Trials', 'Last Trial Outcome']]
# y = df['Product Experience']   # Predicting Product Experience

# # -----------------------
# # Train/Test split
# # -----------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -----------------------
# # Random Forest Classifier
# # -----------------------
# rf = RandomForestClassifier(
#     n_estimators=100,    # number of trees
#     random_state=42,
#     max_depth=5          # prevent overfitting
# )
# rf.fit(X_train, y_train)

# # -----------------------
# # Predictions
# # -----------------------
# y_pred = rf.predict(X_test)

# # -----------------------
# # Evaluation
# # -----------------------
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Feature importance
# importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': rf.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# print("\nFeature Importances:\n", importances)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier, plot_importance

# -------------------------
# Step 1: Load & Prepare Data
# -------------------------
df_large = pd.read_csv('NCT04414150.csv')

X = df_large[['Age', 'Previous Trials', 'Product Experience']]
y = df_large['Last Trial Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------
# Step 2: Train XGBoost Classifier
# -------------------------
model = XGBClassifier(
    n_estimators=200,       # number of trees
    max_depth=3,           # max depth of each tree
    learning_rate=0.1,     # shrinkage step size
    subsample=0.8,         # use 80% of training samples per tree
    colsample_bytree=0.8,  # use 80% of features per tree
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -------------------------
# Step 3: Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("XGBoost Model Evaluation:")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Step 4: Feature Importance
# -------------------------
plt.figure(figsize=(8, 5))
plot_importance(model, importance_type="gain", xlabel="Importance Score")
plt.title("XGBoost Feature Importance (Gain)")
plt.savefig("xgboost_feature_importance.png", dpi=300)
plt.show()

# -------------------------
# Step 5: Save Model
# -------------------------
joblib.dump(model, 'xgb.pkl')
print("\nModel saved as 'trial_outcome_xgb.pkl'")

# -------------------------
# Step 6: Prediction Report
# -------------------------
def generate_prediction_report(model, X_test, y_test):
    results = []
    for i in range(len(X_test)):
        instance = X_test.iloc[i]
        actual = y_test.iloc[i]
        prediction = model.predict([instance])[0]
        results.append({
            'instance': i+1,
            'Age': instance['Age'],
            'Previous Trials': instance['Previous Trials'],
            'Product Experience': instance['Product Experience'],
            'actual_outcome': 'Positive' if actual == 1 else 'Negative',
            'predicted_outcome': 'Positive' if prediction == 1 else 'Negative',
            'correct': 'Yes' if actual == prediction else 'No'
        })
    return pd.DataFrame(results)

report_df = generate_prediction_report(model, X_test, y_test)
print("\nPrediction Report (first 20 rows):")
print("=" * 50)
print(report_df.head(20))


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# Step 1: Prepare Data
# -------------------------
df_large = pd.read_csv('NCT04414150.csv')

X = df_large[['Age', 'Previous Trials', 'Product Experience']]
y = df_large['Last Trial Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------
# Step 2: Train Support Vector Machine
# -------------------------
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Step 3: Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("Model Evaluation:")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Step 4: Confusion Matrix Visualization
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.savefig('svm_confusion_matrix.png', dpi=300)
plt.show()

# -------------------------
# Step 5: Save Model
# -------------------------
joblib.dump(model, 'trial_outcome_svm.pkl')
print("\nModel saved as 'trial_outcome_svm.pkl'")

# -------------------------
# Step 6: Prediction Report
# -------------------------
def generate_prediction_report(model, X_test, y_test):
    results = []
    for i in range(len(X_test)):
        instance = X_test.iloc[i]
        actual = y_test.iloc[i]
        prediction = model.predict([instance])[0]
        results.append({
            'instance': i+1,
            'Age': instance['Age'],
            'Previous Trials': instance['Previous Trials'],
            'Product Experience': instance['Product Experience'],
            'actual_outcome': 'Positive' if actual == 1 else 'Negative',
            'predicted_outcome': 'Positive' if prediction == 1 else 'Negative',
            'correct': 'Yes' if actual == prediction else 'No'
        })
    return pd.DataFrame(results)

report_df = generate_prediction_report(model, X_test, y_test)
print("\nPrediction Report (first 20 rows):")
print("=" * 50)
print(report_df.head(20))
