import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix        
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Veri setini yükleme
try:
    data = pd.read_csv("C:\\Users\\mamie\\Downloads\\train.csv")
    print("Veri yüklendi!")
except FileNotFoundError:
    print("Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()
except Exception as e:
    print(f"Bir hata oluştu: {e}")
    exit()

# Veri sütunlarını kontrol etme
print("Veri sütunları:")
print(data.columns)

# Etiket Dağılımı
if 'Hate' in data.columns:
    print("Etiket Dağılımı:")
    print(data['Hate'].value_counts())
else:
    print("Hate sütunu bulunamadı. Etiket sütunu adını kontrol edin.")
    exit()

# Metin sütununu kontrol etme
if 'Translated Post Description' in data.columns:
    data['text'] = data['Translated Post Description']
elif 'Post description' in data.columns:
    data['text'] = data['Post description']
else:
    print("Metin sütunu bulunamadı. Sütun adlarını kontrol edin.")
    exit()

# Metin uzunluklarını analiz et
data['text_length'] = data['text'].apply(len)
print("Metin Uzunluğu İstatistikleri:")
print(data['text_length'].describe())

# Temizleme Fonksiyonları
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  # Emojilerin Unicode aralıkları
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text, stop_words):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def clean_text(text, stop_words):
    text = text.lower()
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, stop_words)
    return text

# Stopwords seti (İngilizce için)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Temizleme işlemini uygula
data['text_cleaned'] = data['text'].apply(lambda x: clean_text(x, stop_words))

# Eğitim ve Test Verilerini Ayırma
X = data['text_cleaned']
y = data['Hate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SMOTE ile Oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

# Logistic Regression model with balanced class weights
# Alternatif olarak Random Forest veya başka bir model seçilebilir
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_smote, y_train_smote)

# Tahmin ve Değerlendirme
y_pred = model.predict(X_test_tfidf)

# Sınıflandırma Raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print(f"Doğruluk Skoru: {accuracy_score(y_test, y_pred)}")

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()
