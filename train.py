# train.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Scale features -- this is the feature-scaling connection from Phase 0:
# wine's features range from ~0.1 to ~1700, so without scaling, gradient
# descent inside LogisticRegression hits exactly the instability we saw
# with the learning-rate threshold last week.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run():
    max_iter = 1000
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("scaled_features", True)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")
    print(f"Accuracy: {accuracy:.3f}")
