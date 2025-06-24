import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/ASUS/Documents/imdb_dataset/IMDB_Dataset.csv')

documents = df["review"]
labels = df["sentiment"]

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

cleaned_doc = [clean_text(row) for row in documents]

print("İlk 2 temizlenmiş belge:")
print(cleaned_doc[:2])

vectorizer = CountVectorizer()  
x = vectorizer.fit_transform(cleaned_doc[:50])
feature_names = vectorizer.get_feature_names_out()
vector_temsili = x.toarray()

print(f"Vektör temsili şekli: {x.shape}")

word_count = x.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_count))

most_common_5_words = Counter(word_freq).most_common(5)

words, counts = zip(*most_common_5_words)
plt.bar(words, counts)
plt.title("En Sık 5 Kelime")
plt.xlabel("Kelime")
plt.ylabel("Frekans")
plt.show()