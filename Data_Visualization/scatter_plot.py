import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv("../datasets/dataset_train.csv")
#clean the data by removing the first 4 columns
columns_to_keep = ['Hogwarts House','Defense Against the Dark Arts', 'Astronomy']
df_cleaned = df[columns_to_keep]

courses = [col for col in df_cleaned.columns if col != 'Hogwarts House']
houses = df_cleaned['Hogwarts House'].dropna().unique()
colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

df_cleaned.plot(kind='scatter', x=courses[1], y=courses[0], c=df_cleaned['Hogwarts House'].map(colors), alpha=0.25)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=house, markerfacecolor=color, markersize=10) for house, color in colors.items()], title='Hogwarts House')
plt.xlabel(courses[1])
plt.ylabel(courses[0])
plt.show()