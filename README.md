# 🎬 IMDB Review Analysis with Bag of Words

Bu projede, IMDB film yorumları veri seti üzerinde temel doğal dil işleme (NLP) teknikleri uygulanmıştır. Amaç, yorumları temizleyerek Bag of Words yöntemiyle sayısal temsillerini elde etmek ve temel analiz yapmaktır.

## 🧰 Kullanılan Yöntemler

- Metin temizleme (`lowercase`, sayıları kaldırma, noktalama temizleme, stopwords çıkarma)
- `CountVectorizer` ile vektörleme (Bag of Words)
- Kelime frekans analizi (`Counter` ile en sık geçen kelimeler)

## 🗂 Veri Seti

- **Kaynak:** [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Özellikler:**  
  - `review`: Film yorumu (metin)  
  - `sentiment`: Pozitif veya negatif etiket

## 📊 Örnek Çıktı

- İlk 50 yorum vektörleştirildi  
- Vektör matrisi şekli: `(50, kelime_sayısı)`  
- En sık geçen kelimeler: `great`, `movie`, `film`, ...

## 🔜 Gelecek Geliştirme

- TF-IDF ile karşılaştırma yapılabilir  
- `MultinomialNB` ile basit sınıflandırma modeli kurulabilir  
- ROC eğrisi ve f1-score ile performans değerlendirmesi yapılabilir

## 👨‍💻 Geliştirici
**Furkan Yılmaz**  
Yapay zeka ve doğal dil işleme konularında projeler geliştiriyor.
