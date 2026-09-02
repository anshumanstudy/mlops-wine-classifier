import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_data():
    data = load_wine()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)


def train_model(X_train, y_train, max_iter=1000):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return accuracy_score(y_test, predictions)


def main():
    X_train, X_test, y_train, y_test = load_data()
    with mlflow.start_run():
        max_iter = 1000
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("scaled_features", True)

        model, scaler = train_model(X_train, y_train, max_iter)
        accuracy = evaluate(model, scaler, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
