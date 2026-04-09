# helpers/model_trainer.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple, List
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb


class ModelTrainer:
    """
    Entrenador de modelos de regresión para predicción de popularidad en Spotify.
    """
    
    # Grids reducidos y optimizados para 400k+ registros
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
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 X_test: pd.DataFrame,
                 y_train: pd.Series, 
                 y_test: pd.Series,
                 target_col: str = 'popularity',
                 random_state: int = 42, 
                 checkpoint_dir: str = '../temp/checkpoints/'):
        """
        Inicializa el entrenador con datos ya divididos.
        
        Args:
            X_train: Features de entrenamiento
            X_test: Features de test
            y_train: Target de entrenamiento
            y_test: Target de test
            target_col: Nombre de la columna target (para metadata)
            random_state: Semilla
            checkpoint_dir: Directorio para checkpoints
        """
        self.target_col = target_col
        self.checkpoint_dir = checkpoint_dir
        self.random_state = random_state
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.results = {}
        self.best_models = {}
        
        print(f"✓ ModelTrainer inicializado - Target: {target_col}")
        print(f"  Train: {self.X_train.shape[0]:,} muestras, {self.X_train.shape[1]} features")
        print(f"  Test: {self.X_test.shape[0]:,} muestras")
        print(f"  Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")
        print(f"  Target mean: {y_train.mean():.1f} ± {y_train.std():.1f}")
    
    def _get_model(self, model_name: str, **kwargs) -> Any:
        """Retorna instancia del modelo."""
        models = {
            'randomforest': lambda: RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **kwargs),
            'xgboost': lambda: xgb.XGBRegressor(
                random_state=self.random_state, n_jobs=-1, tree_method='hist', **kwargs
            ),
            'catboost': lambda: CatBoostRegressor(random_state=self.random_state, verbose=0, **kwargs),
            'mlp': lambda: MLPRegressor(random_state=self.random_state, early_stopping=True, **kwargs),
            'lightgbm': lambda: lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1, **kwargs)
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(models.keys())}")
        
        return models[model_name]()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas las métricas."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _grid_search_with_kfold(self, 
                                model_name: str, 
                                param_grid: Dict,
                                cv: int, 
                                early_stopping_rounds: Optional[int],
                                val_size: float, 
                                checkpoint_interval: int) -> Tuple[Dict, float]:
        """
        Grid search manual con K-Fold CV y checkpoints.
        Retorna los mejores parámetros y el mejor score de validación.
        """
        from itertools import product
        from sklearn.model_selection import train_test_split
        
        # Generar todas las combinaciones
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        print(f"  Total combinaciones: {len(combinations)}")
        
        best_score = float('inf')
        best_params = None
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for idx, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            
            print(f"\n  [{idx+1}/{len(combinations)}] Probando: {params}")
            
            # CV scores
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
                X_fold_train = self.X_train.iloc[train_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_val = self.y_train.iloc[val_idx]
                
                # Crear modelo con parámetros
                model = self._get_model(model_name, **params)
                
                # Early stopping para modelos que lo soportan
                if early_stopping_rounds and model_name in ['xgboost', 'catboost', 'lightgbm']:
                    # Usar una porción del fold train para early stopping
                    X_tr, X_es, y_tr, y_es = train_test_split(
                        X_fold_train, y_fold_train, 
                        test_size=val_size, random_state=self.random_state
                    )
                    
                    if model_name == 'xgboost':
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_es, y_es)],
                            verbose=False
                        )
                    elif model_name == 'catboost':
                        model.fit(
                            X_tr, y_tr,
                            eval_set=(X_es, y_es),
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )
                    elif model_name == 'lightgbm':
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_es, y_es)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                else:
                    model.fit(X_fold_train, y_fold_train)
                
                # Evaluar en validation fold
                y_pred = model.predict(X_fold_val)
                mse = mean_squared_error(y_fold_val, y_pred)
                fold_scores.append(mse)
            
            # Promedio de scores en CV
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            print(f"    MSE CV: {avg_score:.4f} (±{std_score:.4f})")
            
            # Actualizar mejor modelo
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"    ✓ Nuevo mejor score: {best_score:.4f}")
            
            # Checkpoint cada N iteraciones
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f"{model_name}_checkpoint_{idx+1}.joblib"
                )
                joblib.dump({
                    'best_params': best_params,
                    'best_score': best_score,
                    'iteration': idx + 1
                }, checkpoint_path)
                print(f"Checkpoint guardado: {checkpoint_path}")
        
        return best_params, best_score
    
    def train_model(self, 
                    model_name: str,
                    param_grid: Optional[Dict] = None,
                    cv: int = 5,
                    metric: str = 'mse',
                    early_stopping_rounds: Optional[int] = 50,
                    val_size: float = 0.2,
                    checkpoint_interval: int = 10) -> Dict[str, Any]:
        """
        Entrena un modelo con grid search, K-Fold CV y reentrenamiento final.
        
        Args:
            model_name: 'randomforest', 'xgboost', 'catboost', 'mlp', 'lightgbm'
            param_grid: Grid personalizado o None para usar default
            cv: Número de folds para K-Fold CV
            metric: Métrica para seleccionar mejor modelo (default: 'mse')
            early_stopping_rounds: Rounds para early stopping (None = desactivado)
            val_size: Proporción de datos para early stopping validation
            checkpoint_interval: Cada cuántas iteraciones guardar checkpoint
        
        Returns:
            Diccionario con resultados
        """
        print(f"\n{'='*80}")
        print(f"ENTRENANDO: {model_name.upper()}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        # Grid
        grid = param_grid if param_grid is not None else self.DEFAULT_GRIDS[model_name]
        
        print(f"Grid de búsqueda:")
        for param, values in grid.items():
            print(f"  {param}: {values}")
        
        # Grid search con K-Fold CV
        print(f"\n Buscando mejores hiperparámetros con {cv}-Fold CV...")
        best_params, best_cv_score = self._grid_search_with_kfold(
            model_name, grid, cv, early_stopping_rounds, val_size, checkpoint_interval
        )
        
        print(f"\n✓ Mejores parámetros encontrados: {best_params}")
        print(f"✓ Mejor MSE en CV: {best_cv_score:.4f}")
        
        # Reentrenar en TODO el conjunto de train con los mejores parámetros
        print(f"\n Reentrenando en train completo ({len(self.X_train):,} muestras)...")
        final_model = self._get_model(model_name, **best_params)
        
        if early_stopping_rounds and model_name in ['xgboost', 'catboost', 'lightgbm']:
            from sklearn.model_selection import train_test_split
            # Usar una porción de train para early stopping
            X_tr, X_es, y_tr, y_es = train_test_split(
                self.X_train, self.y_train, test_size=val_size, random_state=self.random_state
            )
            
            if model_name == 'xgboost':
                final_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
            elif model_name == 'catboost':
                final_model.fit(
                    X_tr, y_tr,
                    eval_set=(X_es, y_es),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            elif model_name == 'lightgbm':
                final_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
        else:
            final_model.fit(self.X_train, self.y_train)
        
        print("✓ Reentrenamiento completado")
        
        # Evaluar en test set (evaluación final)
        y_test_pred = final_model.predict(self.X_test)
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred)
        
        # También evaluar en train para ver overfitting
        y_train_pred = final_model.predict(self.X_train)
        train_metrics = self._calculate_metrics(self.y_train, y_train_pred)
        
        print(f"\nMétricas en Train Set:")
        print(f"  MSE: {train_metrics['mse']:.4f}")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  R²: {train_metrics['r2']:.4f}")
        
        print(f"\nMétricas en Test Set (evaluación final):")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"\nTiempo total: {training_time:.1f}s")
        
        # Guardar resultados
        results = {
            'model_name': model_name,
            'best_params': best_params,
            'cv_score': best_cv_score,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'model': final_model
        }
        
        self.results[model_name] = results
        self.best_models[model_name] = final_model
        
        return results
    
    def train_all_models(self, 
                        models: Optional[List[str]] = None,
                        param_grids: Optional[Dict[str, Dict]] = None,
                        cv: int = 5,
                        metric: str = 'mse',
                        early_stopping_rounds: Optional[int] = 50,
                        val_size: float = 0.2) -> Dict[str, Dict]:
        """
        Entrena múltiples modelos.
        
        Args:
            models: Lista de nombres de modelos (None = todos)
            param_grids: Diccionario de grids personalizados por modelo
            cv: Número de folds
            metric: Métrica para optimizar
            early_stopping_rounds: Rounds para early stopping
            val_size: Proporción para validation en early stopping
        
        Returns:
            Diccionario con todos los resultados
        """
        if models is None:
            models = ['randomforest', 'xgboost', 'catboost', 'mlp', 'lightgbm']
        
        param_grids = param_grids or {}
        
        print(f"\n{'#'*80}")
        print(f"ENTRENANDO {len(models)} MODELOS")
        print(f"{'#'*80}")
        
        for model_name in models:
            try:
                grid = param_grids.get(model_name)
                self.train_model(
                    model_name=model_name,
                    param_grid=grid,
                    cv=cv,
                    metric=metric,
                    early_stopping_rounds=early_stopping_rounds,
                    val_size=val_size
                )
            except Exception as e:
                print(f"\n✗ Error entrenando {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def compare_models(self) -> pd.DataFrame:
        """Compara todos los modelos entrenados."""
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for name, res in self.results.items():
            comparison.append({
                'Model': name,
                'CV MSE': res['cv_score'],
                'Train MSE': res['train_metrics']['mse'],
                'Train RMSE': res['train_metrics']['rmse'],
                'Train R²': res['train_metrics']['r2'],
                'Test MSE': res['test_metrics']['mse'],
                'Test RMSE': res['test_metrics']['rmse'],
                'Test MAE': res['test_metrics']['mae'],
                'Test R²': res['test_metrics']['r2'],
                'Time (s)': res['training_time']
            })
        
        df = pd.DataFrame(comparison).sort_values('CV MSE')
        
        print(f"\n{'='*130}")
        print("COMPARACIÓN DE MODELOS")
        print(f"{'='*130}")
        print(df.to_string(index=False))
        
        return df
    
    def get_best_model(self, metric: str = 'mse', dataset: str = 'cv') -> Tuple[str, Any, Dict]:
        """
        Obtiene el mejor modelo.
        
        Args:
            metric: 'mse', 'rmse', 'mae', 'r2'
            dataset: 'cv' (K-Fold CV score), 'train', o 'test'
        """
        if not self.results:
            raise ValueError("No hay modelos entrenados")
        
        if dataset == 'cv':
            metric_getter = lambda x: x['cv_score']
        else:
            metric_key = f'{dataset}_metrics'
            metric_getter = lambda x: x[metric_key][metric]
        
        if metric in ['mse', 'rmse', 'mae']:
            best_name = min(self.results.items(), key=lambda x: metric_getter(x[1]))[0]
        else:  # r2
            best_name = max(self.results.items(), key=lambda x: metric_getter(x[1]))[0]
        
        best_model = self.best_models[best_name]
        best_results = self.results[best_name]
        
        print(f"\n MEJOR MODELO: {best_name.upper()}")
        if dataset == 'cv':
            print(f"   CV MSE: {best_results['cv_score']:.4f}")
        else:
            print(f"   {dataset.upper()} {metric.upper()}: {best_results[f'{dataset}_metrics'][metric]:.4f}")
        
        return best_name, best_model, best_results
    
    
    def save_model(self, model_name: str, path: str) -> None:
        """Guarda modelo y metadata."""
        if model_name not in self.best_models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.best_models[model_name]
        results = self.results[model_name]
        
        joblib.dump(model, path)
        
        metadata = {
            'model_name': model_name,
            'target_col': self.target_col,
            'best_params': results['best_params'],
            'cv_score': results['cv_score'],
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'training_time': results['training_time'],
            'timestamp': datetime.now().isoformat(),
            'n_features': self.X_train.shape[1],
            'n_train_samples': self.X_train.shape[0],
            'n_test_samples': self.X_test.shape[0]
        }
        
        metadata_path = path.replace('.joblib', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        print(f"✓ Modelo guardado: {path}")
        print(f"✓ Metadata guardada: {metadata_path}")
    
    def save_all_models(self, output_dir: str = '../temp/models/') -> None:
        """Guarda todos los modelos."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name in self.best_models.keys():
            path = os.path.join(output_dir, f"{name}.joblib")
            self.save_model(name, path)
        
        # Guardar comparación
        comp = self.compare_models()
        comp.to_csv(os.path.join(output_dir, 'comparison.csv'), index=False)
        
        print(f"\n✓ Todos los modelos guardados en: {output_dir}")


def load_model(path: str) -> Tuple[Any, Dict]:
    """Carga modelo y metadata."""
    model = joblib.load(path)
    metadata = joblib.load(path.replace('.joblib', '_metadata.joblib'))
    
    print(f"✓ Modelo: {metadata['model_name']}")
    print(f"  CV MSE: {metadata['cv_score']:.4f}")
    print(f"  Test R²: {metadata['test_metrics']['r2']:.4f}")
    print(f"  Test RMSE: {metadata['test_metrics']['rmse']:.4f}")
    
    return model, metadata