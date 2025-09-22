import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv("../datasets/dataset_train.csv")
#clean the data by removing the first 4 columns
columns_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
df_cleaned = df.drop(columns=columns_to_drop, axis=1)

# Identifier les matières et les maisons
courses = [col for col in df_cleaned.columns if col != 'Hogwarts House']
houses = df_cleaned['Hogwarts House'].dropna().unique()
colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

#create a histogram for each column in the dataframe
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 15))
axes = axes.flatten()
for i, col in enumerate(courses):
    ax = axes[i]
    ax.set_title(col)
    for house in houses:
        data = df_cleaned[df_cleaned['Hogwarts House'] == house][col].dropna()
        ax.hist(data, alpha=0.8, label=house, color=colors[house])
    ax.set_xlabel('Notes')
    ax.set_ylabel('Nombre d\'élèves')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()
