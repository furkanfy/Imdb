# 🎬 IMDB Film Yorumları Üzerine Metin Analizi

Bu projede, IMDB film yorumları veri seti üzerinde doğal dil işleme (NLP) teknikleri uygulanarak metinler sayısal verilere dönüştürülmüştür. Hem **Bag of Words (BoW)** hem de **TF-IDF** yöntemleri ile vektörleştirme gerçekleştirilmiştir. Amaç, temel metin ön işleme tekniklerini uygulayıp, farklı vektörleme yöntemlerini karşılaştırmaktır.

---

## 📂 Veri Seti

- **Kaynak:** [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Veri:**  
  - `review`: Film yorumu  
  - `sentiment`: Pozitif veya negatif etiket

---

## 🧪 Uygulanan Adımlar

1. **Veri Temizleme:**
   - Küçük harfe çevirme
   - Noktalama ve sayıları kaldırma
   - Stopwords (gereksiz kelimeler) çıkarma
   - Tokenization

2. **Vektörleştirme:**
   - `CountVectorizer` ile Bag of Words (BoW)
   - `TfidfVectorizer` ile TF-IDF

3. **Modelleme (TF-IDF ile):**
   - `MultinomialNB` modeli ile yorum sınıflandırma (pozitif/negatif)

4. **Analiz:**
   - TF-IDF değerlerine göre en anlamlı kelimeler
   - Doğruluk, f1-score gibi değerlendirme metrikleri

---

## ⚖️ Bag of Words vs TF-IDF

### 🟦 Bag of Words (BoW)
- Yalnızca kelimenin kaç kez geçtiğine bakar.
- Örneğin “film çok güzeldi” → “film”: 1, “çok”: 1, “güzeldi”: 1
- Bütün yorumlarda sık geçen kelimeleri **fazla önemli sanabilir**

### 🟩 TF-IDF (Term Frequency-Inverse Document Frequency)
- Kelimenin hem sıklığına hem de tüm belgelerdeki yaygınlığına bakar.
- “film” her yorumda geçiyorsa önemi düşer, ama “büyüleyici” sadece bazı yorumlarda geçiyorsa önemi yükselir.
- Bu yüzden **daha anlamlı kelimelere** dikkat çeker.

### 🎯 Sonuç
TF-IDF, yorumlardaki “daha anlamlı ve ayrıştırıcı” kelimelere ağırlık verdiği için genellikle daha başarılı sonuç verir.

---

## 👨‍💻 Geliştirici

**Furkan Yılmaz**  
Yapay zeka, makine öğrenmesi ve doğal dil işleme alanlarında öğrenmeye ve üretmeye devam ediyor.

---
