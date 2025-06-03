from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np

class SklearnLikeARIMA:
    def __init__(self, order=(9, 1, 2)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.fitted = False
        self.train_index = None
    
    def _reconstruct_dates(self, X):
        if isinstance(X, pd.DataFrame):
            if 'mid_date' in X.columns:
                return pd.to_datetime(X['mid_date'])
            elif all(col in X.columns for col in ['year', 'month', 'day']):
                return pd.to_datetime(dict(
                    year=X['year'],
                    month=X['month'],
                    day=X['day']
                ), errors='coerce')
            elif all(col in X.columns for col in ['year', 'dayofyear']):
                return pd.to_datetime(X['year'] * 1000 + X['dayofyear'], format='%Y%j', errors='coerce')
            else:
                # Fallback: fechas equiespaciadas a partir del último punto de entrenamiento
                return pd.date_range(start=self.train_index[-1], periods=len(X) + 1, freq='MS')[1:]
        else:
            return pd.date_range(start=self.train_index[-1], periods=len(X) + 1, freq='MS')[1:]


    def fit(self, X, y):
        fechas = self._reconstruct_dates(X)
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

        fechas_pred = self._reconstruct_dates(X)
        pred.index = fechas_pred
        return pred.values

        
def get_arima_model(X_train, y_train):
    model = SklearnLikeARIMA(order=(9, 1, 2))
    model.fit(X_train, y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    errors = []

    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        m = SklearnLikeARIMA(order=(9, 1, 2))
        m.fit(X_t, y_t)
        y_pred = m.predict(X_v)
        error = mean_squared_log_error(y_v, np.clip(y_pred, a_min=0, a_max=None))
        errors.append(-error)

    return model, np.array(errors)
