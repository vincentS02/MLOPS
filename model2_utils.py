import joblib

def load_model2():
    model = joblib.load("classification_model.joblib")
    return model

def model2_predict(feature1: float, feature2: float) -> int:
    model = load_model2()
    try:
        # Préparer les données d'entrée pour le modèle
        input_data = [[feature1, feature2]]
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Retourner la prédiction
        return int(prediction[0])
    except Exception as e:
        raise ValueError(f"Erreur dans model2_predict: {e}")
