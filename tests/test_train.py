import numpy as np
from train import load_data, train_model, evaluate


def test_load_data_shapes():
    """Traditional-style test: data has the shape we expect, every time."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 13  # wine dataset always has 13 features
    assert len(y_train) == X_train.shape[0]


def test_data_no_missing_values():
    """Data validation: catches upstream data quality issues before they
    corrupt training. In production this is the check that catches a broken
    upstream feed before it silently trains a bad model."""
    X_train, X_test, y_train, y_test = load_data()
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_data_within_expected_ranges():
    """Data validation: flags incoming data that looks wildly different from
    what the model was trained on -- an early, cheap drift signal, checked
    before the model even runs."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.min() >= 0        # no negative values expected in this dataset
    assert X_train.max() < 2000      # sanity bound (proline feature maxes ~1680)


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
