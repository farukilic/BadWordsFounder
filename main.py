import pandas as pd

############################################################################
""" Bu kısmı satırın en başındaki(virgülden önceki) kısmı alabilmek için yazdım.

df = pd.read_csv("profanity_en.csv")

df['kufurler'] = df['text'].str.split(',').str[0]

yeni_df = pd.DataFrame(df['kufurler'])

yeni_df.to_csv("ilk_satır_alınacak.csv", index=False)
"""
#############################################################################
""" Burada boşluk ve tekrar eden satırları sildim.
df = pd.read_csv("dataset.csv")

df = df.dropna(how="all")
df = df.drop_duplicates(subset="words")

df.to_csv("temizdataset.csv", index= False)

"""
########################################################################
# Burada iki dataseti tek bir datasette topladım.
df = pd.read_csv("labeled_data.csv")

