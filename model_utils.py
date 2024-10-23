import joblib



def load_model():
    model = joblib.load("regression.joblib")
    return model

def model_predict(size: float, nb_rooms: float, garden: str) -> float:
    model = load_model()
    try:
        # Convertir 'garden' en numérique
        garden_numeric = convert_garden_to_numeric(garden)
        
        # Préparer les données d'entrée pour le modèle
        input_data = [[size, nb_rooms, garden_numeric]]
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Retourner la prédiction (supposant que c'est un tableau NumPy)
        return prediction[0]
    except Exception as e:
        raise ValueError(f"Erreur dans model_predict: {e}")

def convert_garden_to_numeric(garden: str) -> int:
    """
    Convertit la valeur de 'garden' en une valeur numérique.
    Par exemple, 'oui' devient 1 et 'non' devient 0.
    """
    true_values = ['oui', 'yes', '1', 'true']
    if isinstance(garden, str):
        return 1 if garden.strip().lower() in true_values else 0
    elif isinstance(garden, (int, float)):
        return 1 if garden == 1 else 0
    else:
        raise ValueError(f"Valeur inattendue pour 'garden': {garden}")