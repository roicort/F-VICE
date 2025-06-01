from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

def get_xgboost_model(X_train, y_train):
    """
    Train an XGBoost model on the provided training data using GridSearchCV (sin lags).
    """
    features = X_train.columns.tolist()

    model = xgb.XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train[features], y_train)
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor score (neg MSE): {grid_search.best_score_}")

    return grid_search.best_estimator_, grid_search.cv_results_