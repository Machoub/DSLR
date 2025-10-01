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
g = sns.pairplot(
    df_cleaned,
    hue='Hogwarts House',
    palette=colors,
    diag_kind='kde',
    plot_kws={'alpha': 0.5}
)

# Supprimer complètement la légende
if g._legend is not None:
    g._legend.remove()

g.set(xticks=[], yticks=[])
plt.tight_layout()
plt.show()