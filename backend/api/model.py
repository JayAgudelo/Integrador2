from helpers.preprocessor import DataPreprocessor
def prediction(preprocessor_route, model, features: dict) -> float:
    """
    Given a preprocessor, model and features dictionary,
    return the predicted popularity score.
    """
    import pandas as pd

    # Convertir a DataFrame
    if 'features' in features:
        features = features['features']
    features_df = pd.DataFrame([features])
    print("DataFrame columns:", features_df.columns.tolist())
    print(features_df)
    preprocessor = DataPreprocessor(features_df)
    preprocessor.load_preprocessor(preprocessor_route)
    # Preprocesar features
    try:
        X_processed =  preprocessor.apply_to_test(features_df,keep_original_names=True)
        print(X_processed)
        # Predecir popularidad
        popularity_pred = model.predict(X_processed)
        
        return float(popularity_pred[0])
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise
