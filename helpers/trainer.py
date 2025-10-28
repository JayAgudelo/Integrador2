import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple, List, Literal
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb


class ModelTrainer:
    """
    Entrenador de modelos de regresión (seleccionando solo uno a la vez).
    """

    DEFAULT_GRIDS = {
        'randomforest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 2]
        },
        'catboost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        },
        'mlp': {
            'hidden_layer_sizes': [(64,), (128,), (128, 64)],
            'activation': ['relu'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['adaptive'],
            'max_iter': [500]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 10],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: Literal["randomforest", "xgboost", "catboost", "mlp", "lightgbm"],
        target_col: str = "popularity",
        test_size: float = 0.2,
        random_state: int = 42,
        checkpoint_dir: str = "../temp/checkpoints/"
    ):
        """
        Inicializa el entrenador y define el modelo a usar.

        Args:
            df: DataFrame con las features y el target.
            model_name: Modelo a entrenar. Opciones:
                        "randomforest", "xgboost", "catboost", "mlp", "lightgbm"
            target_col: Columna objetivo.
            test_size: Proporción del test set.
            random_state: Semilla.
            checkpoint_dir: Carpeta para checkpoints.
        """
        valid_models = list(self.DEFAULT_GRIDS.keys())
        model_name = model_name.lower()

        if model_name not in valid_models:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. "
                f"Opciones válidas: {', '.join(valid_models)}"
            )

        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' no encontrado en el DataFrame.")

        self.model_name = model_name
        self.target_col = target_col
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.results = {}
        self.best_model = None

        print(f"✓ ModelTrainer inicializado")
        print(f"  Modelo seleccionado: {self.model_name.upper()}")
        print(f"  Target: {self.target_col}")
        print(f"  Train: {self.X_train.shape[0]:,} | Test: {self.X_test.shape[0]:,}")

    def _get_model(self, **kwargs) -> Any:
        """Retorna instancia del modelo seleccionado."""
        models = {
            'randomforest': lambda: RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs),
            'xgboost': lambda: xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist', **kwargs),
            'catboost': lambda: CatBoostRegressor(random_state=42, verbose=0, **kwargs),
            'mlp': lambda: MLPRegressor(random_state=42, early_stopping=True, **kwargs),
            'lightgbm': lambda: lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **kwargs)
        }
        return models[self.model_name]()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def _grid_search_with_kfold(self, param_grid: Dict, cv: int, early_stopping_rounds: Optional[int],
                                val_size: float, checkpoint_interval: int) -> Tuple[Dict, float]:
        """Grid search simple con K-Fold."""
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        print(f"  Total combinaciones: {len(combinations)}")

        best_score = float('inf')
        best_params = None
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            fold_scores = []

            for train_idx, val_idx in kfold.split(self.X_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                model = self._get_model(**params)

                if early_stopping_rounds and self.model_name in ['xgboost', 'catboost', 'lightgbm']:
                    X_tr2, X_es, y_tr2, y_es = train_test_split(X_tr, y_tr, test_size=val_size, random_state=42)
                    if self.model_name == 'xgboost':
                        model.fit(X_tr2, y_tr2, eval_set=[(X_es, y_es)], verbose=False)
                    elif self.model_name == 'catboost':
                        model.fit(X_tr2, y_tr2, eval_set=(X_es, y_es),
                                  early_stopping_rounds=early_stopping_rounds, verbose=False)
                    elif self.model_name == 'lightgbm':
                        model.fit(X_tr2, y_tr2, eval_set=[(X_es, y_es)],
                                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
                else:
                    model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                fold_scores.append(mse)

            avg_score = np.mean(fold_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"  ✓ Nuevo mejor MSE: {best_score:.4f} - {params}")

            if (idx + 1) % checkpoint_interval == 0:
                path = os.path.join(self.checkpoint_dir, f"{self.model_name}_chk_{idx+1}.joblib")
                joblib.dump({'best_params': best_params, 'best_score': best_score}, path)
                print(f"  💾 Checkpoint guardado: {path}")

        return best_params, best_score

    def train(self, param_grid: Optional[Dict] = None, cv: int = 5,
              early_stopping_rounds: Optional[int] = 50, val_size: float = 0.2,
              checkpoint_interval: int = 10) -> Dict[str, Any]:
        """
        Entrena el modelo seleccionado usando K-Fold y grid search.
        """
        print(f"\n{'='*70}")
        print(f"Entrenando modelo: {self.model_name.upper()}")
        print(f"{'='*70}")

        start_time = datetime.now()
        grid = param_grid if param_grid is not None else self.DEFAULT_GRIDS[self.model_name]

        best_params, best_cv_score = self._grid_search_with_kfold(
            grid, cv, early_stopping_rounds, val_size, checkpoint_interval
        )

        print(f"\n✓ Mejores parámetros: {best_params}")
        print(f"✓ Mejor MSE CV: {best_cv_score:.4f}")

        final_model = self._get_model(**best_params)
        final_model.fit(self.X_train, self.y_train)
        self.best_model = final_model

        y_pred_train = final_model.predict(self.X_train)
        y_pred_test = final_model.predict(self.X_test)

        results = {
            'best_params': best_params,
            'cv_score': best_cv_score,
            'train_metrics': self._calculate_metrics(self.y_train, y_pred_train),
            'test_metrics': self._calculate_metrics(self.y_test, y_pred_test),
            'training_time': (datetime.now() - start_time).total_seconds(),
            'model': final_model
        }

        self.results = results
        print(f"\n✓ Entrenamiento completado ({results['training_time']:.1f}s)")
        print(f"  Test R²: {results['test_metrics']['r2']:.4f}")

        return results

    def save_model(self, path: str) -> None:
        """Guarda el modelo y metadata."""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        joblib.dump(self.best_model, path)
        meta = {
            'model_name': self.model_name,
            'target_col': self.target_col,
            'best_params': self.results['best_params'],
            'metrics': self.results['test_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(meta, path.replace('.joblib', '_meta.joblib'))
        print(f"✓ Modelo guardado en: {path}")
