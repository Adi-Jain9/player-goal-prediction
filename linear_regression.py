import numpy as np

from data_preprocessing import load_data

def fit_linear_regression(X, y):

    # add a column of 1s so the model can learn the intercept
    ones = np.ones((X.shape[0], 1))

    X_with_intercept = np.column_stack((ones, X))
    
    # find the parameters that minimize error
    parameters, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond = None)

    intercept = parameters[0]

    coefficients = parameters[1:]

    return intercept, coefficients

def predict(X, intercept, coefficients):

    predictions = intercept + X @ coefficients

    return predictions

def mean_absolute_error(actual, predicted):

    errors = np.abs(actual - predicted)

    return np.mean(errors)

def root_mean_squared_error(actual, predicted):

    squared_errors = (actual - predicted) ** 2

    mean_squared_error = np.mean(squared_errors)

    return np.sqrt(mean_squared_error)

def main():
    (
        X_train,
        X_validation,
        X_test,
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        y_train,
        y_validation,
        y_test,
        feature_names,
        means,
        standard_deviations
    ) = load_data()

    # TRAIN MODEL
    intercept, coefficients = fit_linear_regression(X_train_scaled, y_train)

    # MAKE VALIDATION PREDICTIONS
    predictions = predict(X_validation_scaled, intercept, coefficients)

    # CALCULATE ERROR
    mae = mean_absolute_error(y_validation, predictions)

    rmse = root_mean_squared_error(y_validation, predictions)

    print("LINEAR REGRESSION")

    print("Training examples:", len(y_train))
    print("Validation examples:", len(y_validation))

    print()

    print("Coefficients:")

    for name, coefficient in zip(feature_names, coefficients):
        print(f"{name:<35}"f"{coefficient:.6f}")

    print()

    print(f"Validation MAE: {mae:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")

    print()

    print("First 10 predictions:")

    for i in range(min(10, len(y_validation))):
        print(f"Actual: {y_validation[i]:.0f}   "
              f"Predicted: {predictions[i]:.4f}"
              )

if __name__ == "__main__":
    main()
