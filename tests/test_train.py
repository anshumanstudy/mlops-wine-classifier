from train import load_data, train_model, evaluate


def test_load_data_shapes():
    """Traditional-style test: data has the shape we expect, every time."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 13  # wine dataset always has 13 features
    assert len(y_train) == X_train.shape[0]


def test_model_trains_without_error():
    """Traditional-style test: training completes and returns a real model."""
    X_train, X_test, y_train, y_test = load_data()
    model, scaler = train_model(X_train, y_train)
    assert model is not None


def test_model_accuracy_above_threshold():
    """The 'ML test': a moving target, not a fixed assertion.
    This can start failing with zero code changes if future data
    or model changes degrade performance -- that's the point."""
    X_train, X_test, y_train, y_test = load_data()
    model, scaler = train_model(X_train, y_train)
    accuracy = evaluate(model, scaler, X_test, y_test)
    assert accuracy > 0.85
