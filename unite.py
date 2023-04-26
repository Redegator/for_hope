import pandas as pd

# Читаємо файли у відповідні Dataframe
df1 = pd.read_csv('new_RNAs — Nodes.txt', sep='\t', header=None, low_memory=False)
df2 = pd.read_csv('updated NX.txt', header=None, low_memory=False)

# Об'єднуємо дві Dataframe за допомогою методу concat
df_merged = pd.concat([df1, df2], axis=1)

# Записуємо результат до united.txt файлу
df_merged.to_csv('united_new_RNAs — Nodes.txt', sep='\t', index=False, header=None)