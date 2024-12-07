import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB

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
#Dosyayı okuyup ; değeri görünce sütun atlıyor
df = pd.read_csv("dataset_words.csv", sep=";")

#
vektor = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vektor.fit_transform(df['text']).toarray()
y = df['label']


# Veriyi bölüyoruz bir kısmı ile makineyi eğitiyor bir kısmı ile test ediyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)


model = MultinomialNB()
model.fit(X_train, y_train)


# Test et ve sonuçları yazdır
y_pred = model.predict(X_test)
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def check_message(message):
    vectorized = vektor.transform([message]).toarray()
    prediction = model.predict(vectorized)
    if prediction[0] == 1:
        return "Küfür içeriyor!"
    else:
        return "Temiz."


print(check_message(input("Bir kelime giriniz:")))

