import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time

# ===== DATASET YÜKLEME VE HAZIRLAMA =====
print("Oxford-IIIT Pet Dataset Analizi Başlıyor...")
print("=" * 50)

# Dataset yolları
dataset_path = "dataset/images"
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Sınıf isimleri (Oxford-IIIT Pet'te 37 sınıf)
class_names = [
    'Abyssinian', 'American_bulldog', 'American_pit_bull_terrier', 'Basset_hound',
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_shorthair', 'Chihuahua',
    'Egyptian_mau', 'English_cocker_spaniel', 'English_setter', 'German_shorthaired',
    'Great_pyrenees', 'Havanese', 'Japanese_chin', 'Keeshond', 'Leonberger',
    'Maine_coon', 'Miniature_pinscher', 'Newfoundland', 'Persian', 'Pomeranian',
    'Pug', 'Ragdoll', 'Russian_blue', 'Saint_bernard', 'Samoyed', 'Scottish_terrier',
    'Shiba_inu', 'Siamese', 'Sphynx', 'Staffordshire_bull_terrier', 'Wheaten_terrier',
    'Yorkshire_terrier'
]

print(f"Toplam Sınıf Sayısı: {len(class_names)}")
print(f"Kedi Türleri: {len([c for c in class_names if c in ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_shorthair', 'Egyptian_mau', 'Maine_coon', 'Persian', 'Ragdoll', 'Russian_blue', 'Siamese', 'Sphynx']])}")
print(f"Köpek Türleri: {len([c for c in class_names if c not in ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_shorthair', 'Egyptian_mau', 'Maine_coon', 'Persian', 'Ragdoll', 'Russian_blue', 'Siamese', 'Sphynx']])}")

def load_and_preprocess_dataset():
    """Dataset'i yükle ve ön işle"""
    print("\n Dataset yükleniyor...")
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        print(f"  - {class_name} sınıfı işleniyor...")
        
        # Dosya adlarından sınıf isimlerini çıkar
        for filename in os.listdir(dataset_path):
            if filename.startswith(class_name.lower()) and filename.endswith('.jpg'):
                try:
                    img_path = os.path.join(dataset_path, filename)
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize((IMG_SIZE, IMG_SIZE))
                    image_array = np.array(image)
                    
                    images.append(image_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"    Hata: {filename} - {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset yüklendi: {len(images)} resim, {len(np.unique(labels))} sınıf")
    return images, labels

def create_tf_dataset(images, labels, train_split=0.8):
    """TensorFlow dataset oluştur (shuffle ve stratified split ile)"""
    print("\nTensorFlow Dataset oluşturuluyor...")
    
    # Stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_split, random_state=42)
    for train_index, test_index in sss.split(images, labels):
        train_images, test_images = images[train_index], images[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
    
    def preprocess_image(img, label):
        img = tf.cast(img, tf.float32) / 255.0  # Normalizasyon
        return img, label
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000)  #
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")
    return train_ds, test_ds

# ===== FARKLI TRANSFER LEARNING MODELLERİ =====

def create_model(base_model_name, num_classes=37):
    """Farklı base model'lerle model oluştur"""
    
    if base_model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        model_type = "Hafif ve Hızlı"
        
    elif base_model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        model_type = "Derin ve Güçlü"
        
    elif base_model_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        model_type = "Modern ve Verimli"
        
    elif base_model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        model_type = "Klasik ve Güvenilir"
    
    # Base model'i dondur (freeze)
    base_model.trainable = False
    
    # Kendi modelimizi ekle
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, model_type

def train_and_evaluate_model(model, model_name, train_ds, test_ds, epochs=30):
    """Modeli eğit ve değerlendir (daha fazla epoch, düşük learning rate)"""
    print(f"\n {model_name} eğitimi başlıyor...")
    
    # Model derleme
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Eğitim süresini ölç
    start_time = time.time()
    
    # Model eğitimi
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Test performansı
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    
    # Model boyutu
    model_size = model.count_params()
    
    print(f"{model_name} eğitimi tamamlandı!")
    print(f"   - Test Doğruluğu: {test_accuracy*100:.2f}%")
    print(f"   - Eğitim Süresi: {training_time:.2f} saniye")
    print(f"   - Model Boyutu: {model_size:,} parametre")
    
    return {
        'model': model,
        'history': history,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'model_size': model_size
    }

# ===== ANA ANALİZ =====

def main():
    """Ana analiz fonksiyonu"""
    
    # Dataset yükle
    images, labels = load_and_preprocess_dataset()
    train_ds, test_ds = create_tf_dataset(images, labels)
    
    # Test edilecek modeller
    models_to_test = ["MobileNetV2", "ResNet50", "EfficientNetB0", "VGG16"]
    results = {}
    
    print("\n" + "="*60)
    print("TRANSFER LEARNING MODEL KARŞILAŞTIRMASI")
    print("="*60)
    
    for model_name in models_to_test:
        print(f"\n{model_name} analizi...")
        
        # Model oluştur
        model, model_type = create_model(model_name)
        
        print(f"   Model Tipi: {model_type}")
        print(f"   Base Model: {model_name}")
        print(f"   Parametre Sayısı: {model.count_params():,}")
        
        # Modeli eğit ve değerlendir
        result = train_and_evaluate_model(model, model_name, train_ds, test_ds)
        results[model_name] = result
        
        # Modeli kaydet
        model.save(f"oxford_pets_{model_name.lower()}.h5")
        print(f"Model kaydedildi: oxford_pets_{model_name.lower()}.h5")
    
    # ===== SONUÇLARI KARŞILAŞTIR =====
    print("\n" + "="*60)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("="*60)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Doğruluk (%)': result['test_accuracy'] * 100,
            'Eğitim Süresi (s)': result['training_time'],
            'Parametre Sayısı': result['model_size']
        })
    
    # Sonuçları tablo halinde göster
    for data in comparison_data:
        print(f"{data['Model']:15} | "
              f"Doğruluk: {data['Doğruluk (%)']:6.2f}% | "
              f"Süre: {data['Eğitim Süresi (s)']:6.2f}s | "
              f"Parametre: {data['Parametre Sayısı']:8,}")
    
    # En iyi modeli belirle
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\nEN İYİ MODEL: {best_model[0]}")
    print(f"   Doğruluk: {best_model[1]['test_accuracy']*100:.2f}%")
    
    # Production için öneriler
    print("\n" + "="*60)
    print("PRODUCTION ÖNERİLERİ")
    print("="*60)
    print("Mobil Uygulama: MobileNetV2 (hızlı, küçük)")
    print("Web Uygulaması: EfficientNetB0 (denge)")
    print("Yüksek Doğruluk: ResNet50 (güçlü)")
    print("Klasik Yaklaşım: VGG16 (güvenilir)")
    
    return results

if __name__ == "__main__":
    results = main() 