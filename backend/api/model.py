def prediction(preprocessor,model, features: dict) -> float:
    """
    Given a preprocessor, model and features dictionary,
    return the predicted popularity score.
    """
    import pandas as pd
    # Convert features dict to DataFrame
    features_df = pd.DataFrame([features])
    
    # Preprocess features
    X_processed = preprocessor.transform(features_df)
    
    # Predict popularity
    popularity_pred = model.predict(X_processed)
    
    return float(popularity_pred[0])

