import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# Step 1: Prepare Data
# -------------------------
# df_large already exists from earlier code


df_large = pd.read_csv('NCT04414150.csv')

X = df_large[['Age', 'Previous Trials', 'Product Experience']]
y = df_large['Last Trial Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------
# Step 2: Train Decision Tree
# -------------------------
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
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
# Step 4: Visualize Tree
# -------------------------
plt.figure(figsize=(12, 6))
tree.plot_tree(
    model,
    feature_names=['Age', 'Previous Trials', 'Product Experience'],
    class_names=['Negative', 'Positive'],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=10
)
plt.title("Decision Tree for Trial Outcome Prediction")
plt.savefig('decision_tree.png', dpi=300)
plt.show()

# -------------------------
# Step 5: Feature Importance
# -------------------------
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# -------------------------
# Step 6: Save Model
# -------------------------
joblib.dump(model, 'trial_outcome_predictor.pkl')
print("\nModel saved as 'trial_outcome_predictor.pkl'")

# -------------------------
# Step 7: Prediction Report
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
