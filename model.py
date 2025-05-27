from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb

def get_xgboost_model(X_train, y_train):
    """
    Train an XGBoost model on the provided training data.
    
    Parameters:
    - X_train: DataFrame containing the training features.
    - y_train: Series containing the target variable.
    
    Returns:
    - model: Trained XGBoost model.
    """

    # Define the XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train[['year', 'month', 'dayofyear']], y_train)

    tscv = TimeSeriesSplit(n_splits=10)
    scores = cross_val_score(model, X_train[['year', 'month', 'dayofyear']], y_train, cv=tscv, scoring='neg_mean_squared_log_error')
    print(f"Cross-validated neg_mean_squared_log_error: {-scores.mean():.4f} ± {scores.std():.4f}")

    return model, scores