import joblib
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'model.pkl')

metadata = {
    "model_name": "breast_cancer_logreg",
    "threshold_benign": 0.6206578599836922,
    "labels": {
        "0": "malignant",
        "1": "benign"
    },
    "features_count": X.shape[1]
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model created and saved.")