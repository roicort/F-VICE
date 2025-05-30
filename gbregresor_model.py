from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
import numpy as np

def get_gbr_model(X_train, y_train):
    """
    Entrena un modelo Gradient Boosting Regressor con validación cruzada.
    Devuelve el modelo entrenado y los scores.
    """
    model = GradientBoostingRegressor(random_state=42)
    
    scores = cross_val_score(
        model,
        X_train[['year', 'month', 'dayofyear']],
        y_train,
        cv=10,
        scoring='neg_mean_squared_log_error'
    )
    
    model.fit(X_train[['year', 'month', 'dayofyear']], y_train)
    return model, scores
