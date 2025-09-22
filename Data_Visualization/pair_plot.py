import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

df = pd.read_csv("../datasets/dataset_train.csv")
#clean the data by removing the first 4 columns
columns_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
df_cleaned = df.drop(columns=columns_to_drop, axis=1)

# Identifier les matières et les maisons
courses = [col for col in df_cleaned.columns if col != 'Hogwarts House']
houses = df_cleaned['Hogwarts House'].dropna().unique()
colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

sns.set_theme(style="ticks")
pairplot = sns.pairplot(df_cleaned, hue='Hogwarts House', palette=colors, diag_kind='kde', plot_kws={'alpha':0.5})
# 1. Ajuste la position de la grille pour laisser de l'espace à droite (ex: 15% de la figure)
pairplot.fig.subplots_adjust(right=0.85)

# 2. Déplace la légende dans l'espace créé
#    - bbox_to_anchor=(1.05, 0.5) positionne la légende à l'extérieur
#    - loc='center left' ancre la légende par son côté gauche au point défini
pairplot.legend.set_bbox_to_anchor((1.05, 0.5))
pairplot.legend._loc = 'center left'
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()