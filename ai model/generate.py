import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
# Step 2: Train Linear Regression
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Step 3: Predictions
# -------------------------
y_pred_cont = model.predict(X_test)              # continuous predictions
y_pred = np.where(y_pred_cont >= 0.5, 1, 0)      # convert to 0/1

# -------------------------
# Step 4: Evaluation
# -------------------------
print("Model Evaluation:")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nMSE: {mean_squared_error(y_test, y_pred_cont):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_cont):.4f}")

# -------------------------
# Step 5: Coefficients (Feature Importance)
# -------------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Linear Regression Coefficients')
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png', dpi=300)
plt.show()

# -------------------------
# Step 6: Save Model
# -------------------------
joblib.dump(model, 'trial_outcome_regression.pkl')
print("\nModel saved as 'trial_outcome_regression.pkl'")

# -------------------------
# Step 7: Prediction Report
# -------------------------
def generate_prediction_report(model, X_test, y_test, y_pred_cont, y_pred):
    results = []
    for i in range(len(X_test)):
        instance = X_test.iloc[i]
        actual = y_test.iloc[i]
        pred_cont = y_pred_cont[i]
        pred_binary = y_pred[i]
        results.append({
            'instance': i+1,
            'Age': instance['Age'],
            'Previous Trials': instance['Previous Trials'],
            'Product Experience': instance['Product Experience'],
            'actual_outcome': 'Positive' if actual == 1 else 'Negative',
            'predicted_outcome': 'Positive' if pred_binary == 1 else 'Negative',
            'predicted_score': round(pred_cont, 3),
            'correct': 'Yes' if actual == pred_binary else 'No'
        })
    return pd.DataFrame(results)

report_df = generate_prediction_report(model, X_test, y_test, y_pred_cont, y_pred)
print("\nPrediction Report (first 20 rows):")
print("=" * 50)
print(report_df.head(20))
