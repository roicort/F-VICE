from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np

class SklearnLikeARIMA:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.fitted = False
        self.train_index = None

    def fit(self, X, y):
        fechas = pd.to_datetime(X['mid_date'])
        series = pd.Series(y.values, index=fechas).sort_index()

        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        self.fitted = True
        self.train_index = series.index
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Debes entrenar el modelo antes de predecir.")
        
        n_steps = len(X)
        pred = self.fitted_model.forecast(steps=n_steps)

        # Intentar construir fechas para las predicciones
        if isinstance(X, pd.DataFrame):
            if 'mid_date' in X.columns:
                fechas_pred = pd.to_datetime(X['mid_date'])
            elif all(c in X.columns for c in ['year', 'month', 'dayofyear']):
                fechas_pred = pd.to_datetime({
                    'year': X['year'],
                    'month': X['month'],
                    # Aproxima el día como primero del mes por compatibilidad
                    'day': 1
                }, errors='coerce')
            else:
                fechas_pred = pd.date_range(
                    start=self.train_index[-1], periods=n_steps + 1, freq='MS'
                )[1:]
        else:
            fechas_pred = pd.date_range(
                start=self.train_index[-1], periods=n_steps + 1, freq='MS'
            )[1:]

        pred.index = fechas_pred
        return pred.values

def get_arima_model(X_train, y_train):
    model = SklearnLikeARIMA(order=(5, 1, 0))
    model.fit(X_train, y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    errors = []

    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        m = SklearnLikeARIMA(order=(5, 1, 0))
        m.fit(X_t, y_t)
        y_pred = m.predict(X_v)
        error = mean_squared_log_error(y_v, np.clip(y_pred, a_min=0, a_max=None))
        errors.append(-error)

    return model, np.array(errors)
