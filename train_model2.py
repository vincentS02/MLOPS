import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model2():
    # Création d'un dataset exemple
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Création et entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Sauvegarde du modèle
    joblib.dump(model, "classification_model.joblib")

if __name__ == "__main__":
    build_model2()
