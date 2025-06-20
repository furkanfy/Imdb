import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
import nltk

# CSV dosyasını oku
df = pd.read_csv('C:/Users/ASUS/Documents/imdb_dataset/IMDB_Dataset.csv')

# Sütunları al
documents = df["review"]
labels = df["sentiment"]

# Stop words'leri indir ve tanımla
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Metin temizleme fonksiyonu (düzeltilmiş)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Temizlenmiş belgeleri oluştur
cleaned_doc = [clean_text(row) for row in documents]

# İlk 2 temizlenmiş belgeyi yazdır
print("İlk 2 temizlenmiş belge:")
print(cleaned_doc[:2])

# CountVectorizer ile vektörleştirme
vectorizer = CountVectorizer()  # Değişken adını küçük harfle yazdım (standart)
x = vectorizer.fit_transform(cleaned_doc[:50])
feature_names = vectorizer.get_feature_names_out()
vector_temsili = x.toarray()

# Vektör temsili yerine matrisin şeklini yazdır (daha pratik)
print(f"Vektör temsili şekli: {x.shape}")

# Kelime frekanslarını hesapla
word_count = x.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_count))

# En sık 5 kelimeyi bul ve yazdır
most_common_5_words = Counter(word_freq).most_common(5)
print(f"En sık 5 kelime: {most_common_5_words}")