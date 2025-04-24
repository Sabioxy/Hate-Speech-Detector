# Gerekli kütüphaneleri yükleyin
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm import tqdm

# NLTK'den gerekli veri setlerini indir
nltk.download('stopwords')
from nltk.corpus import stopwords

# Veri setini yükleme
data_path = "C:\\Users\\mamie\\Downloads\\hate-speech-and-offensive-language-master\\data\\labeled_data.csv"
df = pd.read_csv(data_path)

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # URL'leri kaldır
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Özel karakterleri kaldır
    text = text.strip()
    return text

# Metin temizleme uygulama
df['tweet'] = df['tweet'].apply(clean_text)

# Sınıf dağılımını dengeleme (oversampling)
class_counts = df['class'].value_counts()
df_balanced = pd.concat([df[df['class'] == cls].sample(class_counts.max(), replace=True) for cls in class_counts.index])
df = df_balanced.reset_index(drop=True)

# Metinleri ve etiketleri ayırma
texts = df['tweet'].values
labels = df['class'].values

# Bert Tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset sınıfı
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Parametreler
EPOCHS = 10
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Veri setini eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# DataLoader oluşturma
train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model yükleme
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, hidden_dropout_prob=0.4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Sınıf ağırlıkları
class_counts = df['class'].value_counts().sort_index().values
class_weights = torch.tensor([sum(class_counts) / c for c in class_counts]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Optimizasyon
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

# Early Stopping için parametreler
best_val_loss = float('inf')
early_stopping_patience = 3
early_stopping_counter = 0

# Eğitim fonksiyonu
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(data_loader, leave=True, desc="Eğitim")
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # İlerleme çubuğunu güncelle
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    return total_loss / len(data_loader), 100 * correct / total

# Değerlendirme fonksiyonu
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    loop = tqdm(data_loader, leave=True, desc="Doğrulama")
    
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # İlerleme çubuğunu güncelle
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    return total_loss / len(data_loader), 100 * correct / total, all_preds, all_labels

# Eğitim döngüsü
accuracies = []
history = {'train_acc': [], 'val_acc': []}
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
    print(f"Eğitim Kaybı: {train_loss:.4f}, Eğitim Doğruluğu: {train_acc:.2f}%")

    val_loss, val_acc, val_preds, val_labels = eval_model(model, test_loader, loss_fn, device)
    print(f"Doğrulama Kaybı: {val_loss:.4f}, Doğrulama Doğruluğu: {val_acc:.2f}%")
    accuracies.append((epoch + 1, val_acc))
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(classification_report(val_labels, val_preds, target_names=['Hate Speech', 'Offensive', 'Neutral']))

    # Early Stopping kontrolü
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Erken durdurma tetiklendi.")
        break

# En iyi modeli yükleme
model.load_state_dict(torch.load("best_model.pt"))

# Karışıklık matrisi
def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Hate Speech', 'Offensive', 'Neutral'], yticklabels=['Hate Speech', 'Offensive', 'Neutral'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karışıklık Matrisi')
    plt.show()

# Test sonuçları
val_loss, val_acc, val_preds, val_labels = eval_model(model, test_loader, loss_fn, device)
print(f"Test Kaybı: {val_loss:.4f}, Test Doğruluğu: {val_acc:.2f}%")
plot_confusion_matrix(val_preds, val_labels)

# Model geçmişi görselleştirme
def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Eğitim Doğruluğu')
    plt.plot(history['val_acc'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk (%)')
    plt.title('Model Eğitimi ve Doğrulama Doğruluğu')
    plt.legend()
    plt.show()

plot_history(history)
