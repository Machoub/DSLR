import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def _sigmoid(self, x): # cest le g(z) = 1 / (1 + e^(-z)) qui est le h
        return 1 / (1 + np.exp(-x))
    
    def cost_function(self, h, y):
        m = len(y)
        return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self.cost_function(h, y)
            self.cost_history.append(cost)

    def score(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        h = self._sigmoid(linear_model)
        return np.where(h >= 0.5, 1, 0)
    
    def get_weights(self):
        return self.weights, self.bias

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <data.csv>")
        sys.exit(1)

    try:
        data = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: file not found at {sys.argv[1]}")
        sys.exit(1)

    data = data.dropna()
    selected_features = [
        'Herbology', 
        'Defense Against the Dark Arts',
        'Divination',
        'Ancient Runes',
        'Flying',
        'Muggle Studies'
    ]
    features = data[selected_features]
    houses = data['Hogwarts House'].unique()
    #std the features
    # Créer et entraîner le scaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Utiliser le MÊME scaler pour transformer les données de test
    all_weights = {}

    for house in houses:
        x_train, x_test, y_train, y_test = train_test_split(features, data['Hogwarts House'], test_size=0.2, random_state=42)
        y_train = (y_train == house).astype(int)
        y_test = (y_test == house).astype(int)
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(x_train, y_train)
        predt = model.score(x_test)
        accuracy = np.mean(predt == y_test) * 100
        print(f"Accuracy for {house}: {accuracy}%")
        weights, bias = model.get_weights()
        all_weights[house] = (weights, bias)

    print("All weights and biases:")
    for house, (weights, bias) in all_weights.items():
        print(f"{house}: weights={weights}, bias={bias}")

    # Sauvegarder les poids dans un fichier texte
    with open("weights.txt", "w") as f:
        # Inclure les noms des caractéristiques pour référence future
        f.write(f"features:{','.join(selected_features)}\n")
        f.write("---\n")
        for house, (weights, bias) in all_weights.items():
            # Convertir le tableau de poids en une chaîne de caractères
            weights_str = ",".join(map(str, weights))
            f.write(f"{house}\n")
            f.write(f"weights:{weights_str}\n")
            f.write(f"bias:{bias}\n")
            f.write("---\n") # Séparateur pour la lisibilité

    print("\nTraining complete. Weights saved to weights.txt")

if __name__ == "__main__":
    main()