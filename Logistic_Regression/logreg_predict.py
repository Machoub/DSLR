import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_weights_from_txt(filepath):
    """Charge les poids et les caractéristiques depuis un fichier .txt personnalisé."""
    all_weights = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # La première ligne contient les caractéristiques
            features_line = lines[0].strip()
            if features_line.startswith('features:'):
                selected_features = features_line.split(':')[1].split(',')
            else:
                raise ValueError("Le fichier de poids ne contient pas la ligne 'features:'.")

            current_house = None
            for line in lines[2:]: # Ignorer les deux premières lignes (features et '---')
                line = line.strip()
                if not line or line == '---':
                    current_house = None # Réinitialiser pour la prochaine maison
                    continue
                
                if ':' in line: # Vérifier si la ligne contient une clé et une valeur
                    key, value = line.split(':', 1)
                    if current_house:
                        if key == 'weights':
                            # Gérer le cas où la valeur est vide après le split
                            weights_list = value.split(',') if value else []
                            weights = np.array([float(w) for w in weights_list])
                            all_weights[current_house]['weights'] = weights
                        elif key == 'bias':
                            bias = float(value) if value else 0.0
                            all_weights[current_house]['bias'] = bias
                else: # C'est le nom de la maison
                    current_house = line
                    all_weights[current_house] = {}
        return all_weights, selected_features
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de poids : {e}")
        sys.exit(1)

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_to_predict.csv> <weights.txt>")
        sys.exit(1)

    # 1. Charger les poids et les noms des caractéristiques
    all_weights, selected_features = load_weights_from_txt(sys.argv[2])

    # 2. Charger et préparer les données de test
    try:
        data = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: file not found at {sys.argv[1]}")
        sys.exit(1)

    # Conserver l'index original pour la sortie
    original_index = data['Index']
    features = data[selected_features]

    # Remplir les valeurs manquantes avec la moyenne de chaque colonne
    for col in features.columns:
        features[col].fillna(features[col].mean(), inplace=True)

    # Normaliser les données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 3. Faire les prédictions
    predictions = []
    for instance in features_scaled:
        scores = {}
        for house, params in all_weights.items():
            weights = params['weights']
            bias = params['bias']
            linear_model = np.dot(instance, weights) + bias
            probability = _sigmoid(linear_model)
            scores[house] = probability
        
        # Choisir la maison avec la plus haute probabilité
        predicted_house = max(scores, key=scores.get)
        predictions.append(predicted_house)

    # 4. Sauvegarder les résultats
    output_df = pd.DataFrame({
        'Index': original_index,
        'Hogwarts House': predictions
    })
    output_df.to_csv('houses.csv', index=False)
    print("Predictions saved to houses.csv")

if __name__ == "__main__":
    main()