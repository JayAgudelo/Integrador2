import joblib
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    """
    Clase para preprocesamiento de datos con seguimiento de transformaciones.
    Soporta fit en train y transform en test sin data leakage.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el preprocesador con un DataFrame.
        
        Args:
            df: DataFrame a procesar
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.transformations = []
        
        # Columnas por defecto
        self.numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                            'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                            'tempo', 'time_signature', 'valence', 'duration_ms']
        self.cat_cols = ['genre']
        self.preprocessor = None
        
        # Variables para guardar transformaciones aprendidas
        self.top_genres_ = None
        self.log_transform_cols_ = None
        self.log_transform_constant_ = None
        self.dropped_cols_ = None
        self.drop_nulls_subset_ = None
        self.drop_nulls_threshold_ = None
        self.remove_duplicates_subset_ = None
        
        print(f"Preprocesador inicializado con {self.df.shape[0]} filas y {self.df.shape[1]} columnas")
    
    def get_info(self) -> None:
        """Muestra información general del DataFrame actual."""
        print("\n=== INFORMACIÓN DEL DATASET ===")
        print(f"Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
        print(f"Dimensiones originales: {self.original_shape[0]} filas x {self.original_shape[1]} columnas")
        print(f"\nColumnas: {list(self.df.columns)}")
        print(f"\nTipos de datos:\n{self.df.dtypes}")
        print(f"\nValores nulos por columna:\n{self.df.isnull().sum()}")
        print(f"\nMemoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if self.transformations:
            print(f"\n=== TRANSFORMACIONES APLICADAS ===")
            for i, trans in enumerate(self.transformations, 1):
                print(f"{i}. {trans}")
    
    def drop_columns(self, columns: Union[List[str], str]) -> 'DataPreprocessor':
        """
        Elimina columnas del DataFrame.
        
        Args:
            columns: Lista de nombres de columnas o nombre de columna individual
        
        Returns:
            self para permitir method chaining
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Verificar que las columnas existen
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            print(f"Advertencia: Las siguientes columnas no existen: {missing_cols}")
        
        existing_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        
        # GUARDAR: columnas eliminadas
        self.dropped_cols_ = existing_cols
        
        transformation = f"Eliminadas {len(existing_cols)} columnas: {existing_cols}"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        print(f"  Forma resultante: {self.df.shape}")
        
        return self
    
    def filter_top_genres(self, 
                         genre_col: str = 'genre', 
                         popularity_col: str = 'popularity', 
                         top_n: int = 20) -> 'DataPreprocessor':
        """
        Filtra el DataFrame para mantener solo los top N géneros más populares.
        
        Args:
            genre_col: Nombre de la columna de género
            popularity_col: Nombre de la columna de popularidad
            top_n: Número de géneros top a mantener
        
        Returns:
            self para permitir method chaining
        """
        # Verificar que las columnas existen
        if genre_col not in self.df.columns:
            raise ValueError(f"La columna '{genre_col}' no existe en el DataFrame")
        if popularity_col not in self.df.columns:
            raise ValueError(f"La columna '{popularity_col}' no existe en el DataFrame")
        
        original_rows = len(self.df)
        original_genres = self.df[genre_col].nunique()
        
        # Calcular la popularidad promedio por género
        genre_popularity = self.df.groupby(genre_col)[popularity_col].mean().sort_values(ascending=False)
        
        # Obtener los top N géneros
        top_genres = genre_popularity.head(top_n).index.tolist()
        
        # GUARDAR: géneros seleccionados
        self.top_genres_ = top_genres
        
        print(f"\n=== FILTRANDO TOP {top_n} GÉNEROS ===")
        print(f"Top {top_n} géneros por popularidad promedio:")
        for i, genre in enumerate(top_genres, 1):
            avg_pop = genre_popularity[genre]
            count = len(self.df[self.df[genre_col] == genre])
            print(f"  {i}. {genre}: {avg_pop:.2f} (n={count})")
        
        # Filtrar el DataFrame
        self.df = self.df[self.df[genre_col].isin(top_genres)].copy()
        
        rows_removed = original_rows - len(self.df)
        transformation = f"Filtrado top {top_n} géneros: {original_rows} → {len(self.df)} filas ({rows_removed} eliminadas, {rows_removed/original_rows*100:.2f}%)"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        
        return self
    
    def drop_nulls(self, columns: Optional[Union[List[str], str]] = None, 
                   threshold: Optional[float] = None) -> 'DataPreprocessor':
        """
        Elimina filas con valores nulos.
        
        Args:
            columns: Columnas específicas a revisar (None = todas)
            threshold: Porcentaje mínimo de valores no nulos para mantener fila (0-1)
        
        Returns:
            self para permitir method chaining
        """
        original_rows = len(self.df)
        
        # GUARDAR: parámetros de drop_nulls
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            self.drop_nulls_subset_ = columns
            self.df = self.df.dropna(subset=columns)
            desc = f"en columnas {columns}"
        elif threshold:
            self.drop_nulls_threshold_ = threshold
            self.df = self.df.dropna(thresh=int(threshold * len(self.df.columns)))
            desc = f"con threshold {threshold}"
        else:
            self.drop_nulls_subset_ = None
            self.drop_nulls_threshold_ = None
            self.df = self.df.dropna()
            desc = "en todas las columnas"
        
        rows_removed = original_rows - len(self.df)
        transformation = f"Eliminadas {rows_removed} filas con nulos {desc}"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Elimina filas duplicadas.
        
        Args:
            subset: Columnas a considerar para duplicados (None = todas)
        
        Returns:
            self para permitir method chaining
        """
        original_rows = len(self.df)
        
        # GUARDAR: subset de remove_duplicates
        self.remove_duplicates_subset_ = subset
        
        self.df = self.df.drop_duplicates(subset=subset)
        
        rows_removed = original_rows - len(self.df)
        desc = f"basado en {subset}" if subset else "en todas las columnas"
        transformation = f"Eliminados {rows_removed} duplicados {desc}"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        
        return self

    def log_transform(self, 
                     columns: Optional[List[str]] = None,
                     default_cols: Optional[List[str]] = None,
                     add_constant: float = 1e-10) -> 'DataPreprocessor':
        """
        Aplica transformación logarítmica a columnas especificadas.
        Útil para reducir el sesgo en distribuciones asimétricas.
        
        Args:
            columns: Lista de columnas a transformar (si None, usa default_cols)
            default_cols: Columnas por defecto si columns es None
            add_constant: Constante pequeña para evitar log(0)
        
        Returns:
            self para permitir method chaining
        """
        # Definir columnas por defecto si no se especifican
        if columns is None:
            if default_cols is None:
                default_cols = ['instrumentalness', 'liveness', 'speechiness', 'duration_ms']
            columns = default_cols
        
        # Verificar que las columnas existen
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            print(f"Advertencia: Columnas no encontradas: {missing_cols}")
            columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("No hay columnas válidas para transformar")
            return self
        
        # GUARDAR: columnas y constante de log transform
        self.log_transform_cols_ = columns
        self.log_transform_constant_ = add_constant
        
        print(f"\n=== APLICANDO TRANSFORMACIÓN LOGARÍTMICA ===")
        print(f"Columnas: {columns}")
        
        for col in columns:
            # Verificar valores negativos
            if (self.df[col] < 0).any():
                print(f"Advertencia: {col} contiene valores negativos. Se tomarán valores absolutos.")
                self.df[col] = self.df[col].abs()
            
            # Mostrar estadísticas antes
            skew_before = self.df[col].skew()
            
            # Aplicar log(x + constant)
            self.df[col] = np.log(self.df[col] + add_constant)
            
            # Mostrar estadísticas después
            skew_after = self.df[col].skew()
            
            print(f"  {col}: skew antes={skew_before:.3f}, después={skew_after:.3f}")
        
        transformation = f"Aplicada transformación logarítmica a {len(columns)} columnas: {columns}"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        
        return self

    def transform(self, 
                 numeric_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 keep_original_names: bool = False,ordinal_encoder = False) -> 'DataPreprocessor':
        """
        Aplica StandardScaler a columnas numéricas y OneHotEncoder a columnas categóricas usando Pipeline.
        
        Args:
            numeric_cols: Lista de columnas numéricas (usa self.numeric_cols si es None)
            cat_cols: Lista de columnas categóricas (usa self.cat_cols si es None)
            keep_original_names: Si True, intenta mantener nombres de columnas legibles
        
        Returns:
            self para permitir method chaining
        """
        from sklearn.pipeline import Pipeline
        
        # Usar columnas por defecto si no se especifican
        numeric_cols = numeric_cols if numeric_cols is not None else self.numeric_cols
        cat_cols = cat_cols if cat_cols is not None else self.cat_cols
        
        # Verificar que las columnas existen en el DataFrame
        missing_numeric = [col for col in numeric_cols if col not in self.df.columns]
        missing_cat = [col for col in cat_cols if col not in self.df.columns]
        
        if missing_numeric:
            print(f"Advertencia: Columnas numéricas no encontradas: {missing_numeric}")
            numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if missing_cat:
            print(f"Advertencia: Columnas categóricas no encontradas: {missing_cat}")
            cat_cols = [col for col in cat_cols if col in self.df.columns]
        
        if not numeric_cols and not cat_cols:
            raise ValueError("No hay columnas válidas para transformar")
        
        print(f"\n=== APLICANDO TRANSFORMACIONES ===")
        print(f"Columnas numéricas ({len(numeric_cols)}): {numeric_cols}")
        print(f"Columnas categóricas ({len(cat_cols)}): {cat_cols}")
        
        # Guardar columnas que no se van a transformar
        other_cols = [col for col in self.df.columns if col not in numeric_cols + cat_cols]
        df_other = self.df[other_cols].copy() if other_cols else None
        
        # Crear pipelines para cada tipo de columna
        transformers = []
        
        if numeric_cols:
            numeric_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, numeric_cols))
        
        if cat_cols:
            if ordinal_encoder:
                categorical_pipeline = Pipeline([
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
            else:
                categorical_pipeline = Pipeline([
                    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ])
            transformers.append(('cat', categorical_pipeline, cat_cols))
        
        # Crear ColumnTransformer con los pipelines
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Transformar los datos
        transformed_data = self.preprocessor.fit_transform(self.df)
        
        # Obtener nombres de columnas
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Limpiar nombres si se solicita
        if keep_original_names:
            feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
        
        # Crear nuevo DataFrame con datos transformados
        df_transformed = pd.DataFrame(
            transformed_data,
            columns=feature_names,
            index=self.df.index
        )
        
        # Agregar columnas que no se transformaron
        if df_other is not None:
            df_transformed = pd.concat([df_transformed, df_other], axis=1)
        
        self.df = df_transformed
        
        transformation = f"Aplicado StandardScaler a {len(numeric_cols)} columnas numéricas y OneHotEncoder a {len(cat_cols)} columnas categóricas"
        self.transformations.append(transformation)
        print(f"✓ {transformation}")
        print(f"  Forma resultante: {self.df.shape}")
        print(f"  Nuevas columnas: {len(feature_names)} features")
        
        return self
    
    def apply_to_test(self, test_df: pd.DataFrame, keep_original_names: bool = False) -> pd.DataFrame:
        """
        Aplica TODAS las transformaciones aprendidas en train a un conjunto de test.
        
        Args:
            test_df: DataFrame de test con columnas originales
            keep_original_names: Si True, limpia nombres de columnas
            
        Returns:
            DataFrame de test transformado
        """
        print("\n" + "="*80)
        print("APLICANDO TRANSFORMACIONES A TEST SET")
        print("="*80)
        
        df_test = test_df.copy()
        original_rows = len(df_test)
        
        # 1. Drop nulls
        if self.drop_nulls_subset_ is not None:
            df_test = df_test.dropna(subset=self.drop_nulls_subset_)
            print(f"✓ Drop nulls (subset={self.drop_nulls_subset_}): {original_rows} → {len(df_test)}")
        elif self.drop_nulls_threshold_ is not None:
            df_test = df_test.dropna(thresh=int(self.drop_nulls_threshold_ * len(df_test.columns)))
            print(f"✓ Drop nulls (threshold={self.drop_nulls_threshold_}): {original_rows} → {len(df_test)}")
        elif hasattr(self, 'drop_nulls_subset_'):  # Se llamó drop_nulls() sin parámetros
            df_test = df_test.dropna()
            print(f"✓ Drop nulls (all): {original_rows} → {len(df_test)}")
        
        # 2. Remove duplicates
        if self.remove_duplicates_subset_ is not None or hasattr(self, 'remove_duplicates_subset_'):
            original_rows = len(df_test)
            df_test = df_test.drop_duplicates(subset=self.remove_duplicates_subset_)
            print(f"✓ Remove duplicates: {original_rows} → {len(df_test)}")
        
        # 3. Drop columns
        if self.dropped_cols_:
            existing = [col for col in self.dropped_cols_ if col in df_test.columns]
            if existing:
                df_test = df_test.drop(columns=existing)
                print(f"✓ Dropped columns: {existing}")
        
        # 4. Filter top genres
        if self.top_genres_:
            original_rows = len(df_test)
            df_test = df_test[df_test['genre'].isin(self.top_genres_)]
            removed = original_rows - len(df_test)
            print(f"✓ Filter genres (usando {len(self.top_genres_)} géneros de train): {original_rows} → {len(df_test)} ({removed} eliminados)")
        
        # 5. Log transform
        if self.log_transform_cols_:
            for col in self.log_transform_cols_:
                if col in df_test.columns:
                    # Aplicar abs si hay negativos
                    if (df_test[col] < 0).any():
                        df_test[col] = df_test[col].abs()
                    df_test[col] = np.log(df_test[col] + self.log_transform_constant_)
            print(f"✓ Log transform aplicado a: {self.log_transform_cols_}")
        
        # 6. Apply preprocessor (StandardScaler + OneHotEncoder)
        if self.preprocessor is None:
            raise ValueError("Preprocessor no entrenado. Ejecuta transform() en train primero.")
        
        transformed_data = self.preprocessor.transform(df_test)
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Limpiar nombres si se solicita
        if keep_original_names:
            feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
        
        df_transformed = pd.DataFrame(
            transformed_data,
            columns=feature_names,
            index=df_test.index
        )
        
        print(f"✓ StandardScaler + OneHotEncoder aplicado")
        print(f"\n{'='*80}")
        print(f"TEST SET TRANSFORMADO: {df_transformed.shape}")
        print(f"{'='*80}")
        
        return df_transformed
    
    def save(self, output_path: str, **kwargs) -> None:
        """
        Guarda el DataFrame procesado.
        
        Args:
            output_path: Ruta donde guardar el archivo
            **kwargs: Argumentos adicionales para to_csv
        """
        try:
            self.df.to_csv(output_path, index=False, **kwargs)
            print(f"✓ Datos guardados exitosamente en: {output_path}")
        except Exception as e:
            print(f"✗ Error al guardar el archivo: {e}")
            raise
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame procesado.
        
        Returns:
            DataFrame procesado
        """
        return self.df
    
    def save_preprocessor(self, path: str) -> None:
        """
        Guarda el preprocesador completo.
        
        Args:
            path: Ruta donde guardar (con extensión .joblib)
        """
        if self.preprocessor is None:
            print("Advertencia: No hay preprocesador para guardar. Ejecuta transform() primero.")
            return
        
        try:
            # Guardar el preprocesador
            joblib.dump(self.preprocessor, path)
            print(f"✓ Preprocesador guardado en: {path}")
            
            # Guardar metadata completa
            metadata_path = path.replace('.joblib', '_metadata.joblib')
            metadata = {
                'numeric_cols': self.numeric_cols,
                'cat_cols': self.cat_cols,
                'original_shape': self.original_shape,
                'transformations': self.transformations,
                'final_shape': self.df.shape,
                'feature_names': list(self.df.columns),
                # Guardar transformaciones aprendidas
                'top_genres_': self.top_genres_,
                'log_transform_cols_': self.log_transform_cols_,
                'log_transform_constant_': self.log_transform_constant_,
                'dropped_cols_': self.dropped_cols_,
                'drop_nulls_subset_': self.drop_nulls_subset_,
                'drop_nulls_threshold_': self.drop_nulls_threshold_,
                'remove_duplicates_subset_': self.remove_duplicates_subset_
            }
            joblib.dump(metadata, metadata_path)
            print(f"✓ Metadata guardada en: {metadata_path}")
            
        except Exception as e:
            print(f"✗ Error al guardar preprocesador: {e}")
            raise

    def load_preprocessor(self, path: str) -> 'DataPreprocessor':
        """
        Carga un preprocesador guardado previamente.
        
        Args:
            path: Ruta del preprocesador guardado (con extensión .joblib)
        
        Returns:
            self para permitir method chaining
        """
        try:
            # Cargar preprocesador
            self.preprocessor = joblib.load(path)
            print(f"✓ Preprocesador cargado desde: {path}")
            
            # Cargar metadata
            metadata_path = path.replace('.joblib', '_metadata.joblib')
            metadata = joblib.load(metadata_path)
            
            self.numeric_cols = metadata['numeric_cols']
            self.cat_cols = metadata['cat_cols']
            self.original_shape = metadata.get('original_shape', (0, 0))
            
            # Cargar transformaciones aprendidas
            self.top_genres_ = metadata.get('top_genres_')
            self.log_transform_cols_ = metadata.get('log_transform_cols_')
            self.log_transform_constant_ = metadata.get('log_transform_constant_')
            self.dropped_cols_ = metadata.get('dropped_cols_')
            self.drop_nulls_subset_ = metadata.get('drop_nulls_subset_')
            self.drop_nulls_threshold_ = metadata.get('drop_nulls_threshold_')
            self.remove_duplicates_subset_ = metadata.get('remove_duplicates_subset_')
            
            print(f"✓ Metadata cargada")
            print(f"  Columnas numéricas: {self.numeric_cols}")
            print(f"  Columnas categóricas: {self.cat_cols}")
            if self.top_genres_:
                print(f"  Top géneros aprendidos: {len(self.top_genres_)}")
            
        except Exception as e:
            print(f"✗ Error al cargar preprocesador: {e}")
            raise
        
        return self
