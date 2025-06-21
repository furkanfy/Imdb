import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

# CSV dosyasını oku
df = pd.read_csv('C:/Users/ASUS/Documents/imdb_dataset/IMDB_Dataset.csv')

# Sütunları al
documents = df["review"]
labels = df["sentiment"]

# Stop words'leri indir ve tanımla
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Metin temizleme fonksiyonu
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

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)  # En sık geçen 5000 kelime
X = vectorizer.fit_transform(cleaned_doc)  # Doğru veri: cleaned_doc
print("TF-IDF shape:", X.shape)

# TF-IDF matrisini yoğun forma çevir
tfidf_array = X.toarray()

# Kelimelerin toplam TF-IDF skorlarını hesapla
sums = np.sum(tfidf_array, axis=0)  # Toplam TF-IDF skorları
top_words_idx = sums.argsort()[::-1][:10]  # En yüksek 10 kelime indeksi
top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
top_scores = sums[top_words_idx]

# En anlamlı kelimeleri yazdır
print("En anlamlı kelimeler (TF-IDF'e göre):", top_words)

# Görselleştirme: En anlamlı 10 kelimeyi çubuk grafikle göster
plt.figure(figsize=(10, 5))
plt.bar(top_words, top_scores, color='skyblue')
plt.title('En Anlamlı 10 Kelime (TF-IDF Skorlarına Göre)')
plt.xlabel('Kelime')
plt.ylabel('Toplam TF-IDF Skoru')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()