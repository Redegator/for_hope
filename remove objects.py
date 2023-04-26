import pandas as pd

Survival = pd.read_csv(r'C:\Users\redegator\PycharmProjects\hope\RNAs — Nodes.txt', delimiter = '\t')

Survival = Survival.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id'])
Survival.to_csv("new_RNAs — Nodes.txt", sep='\t', index=False)
