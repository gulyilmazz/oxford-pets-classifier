import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# === Otomatik Train/Val Split (Eğer yoksa) ===
def prepare_train_val_split(image_dir="dataset/images", train_dir="data/train", val_dir="data/val", val_ratio=0.2, random_state=42):
    """dataset/images klasöründen stratified şekilde data/train ve data/val klasörlerini oluşturur."""
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("[Split] Train/val klasörleri zaten var, atlanıyor.")
        return
    print("[Split] Train/val klasörleri oluşturuluyor...")
    # Tüm dosyaları ve sınıf isimlerini topla
    files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    class_names = ['_'.join(f.split('_')[:-1]).lower() for f in files]
    files = np.array(files)
    class_names = np.array(class_names)
    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    for train_idx, val_idx in sss.split(files, class_names):
        train_files, val_files = files[train_idx], files[val_idx]
    # Klasörleri oluştur ve dosyaları kopyala
    for split, split_files in zip([train_dir, val_dir], [train_files, val_files]):
        for f in split_files:
            class_name = '_'.join(f.split('_')[:-1]).lower()
            class_dir = os.path.join(split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy2(os.path.join(image_dir, f), os.path.join(class_dir, f))
    print(f"[Split] Train: {len(train_files)}, Val: {len(val_files)}")

# === Train/Val Split Hazırlığı ===
prepare_train_val_split()

# ===== PRODUCTION-READY MODEL SINIFI =====
class OxfordPetsClassifier:
    """Production-ready Oxford-IIIT Pet sınıflandırıcı"""
    
    def __init__(self, model_name="EfficientNetB0", img_size=224):
        self.model_name = model_name
        self.img_size = img_size
        self.model = None
        self.class_names = [
            'Abyssinian', 'American_bulldog', 'American_pit_bull_terrier', 'Basset_hound',
            'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_shorthair', 'Chihuahua',
            'Egyptian_mau', 'English_cocker_spaniel', 'English_setter', 'German_shorthaired',
            'Great_pyrenees', 'Havanese', 'Japanese_chin', 'Keeshond', 'Leonberger',
            'Maine_coon', 'Miniature_pinscher', 'Newfoundland', 'Persian', 'Pomeranian',
            'Pug', 'Ragdoll', 'Russian_blue', 'Saint_bernard', 'Samoyed', 'Scottish_terrier',
            'Shiba_inu', 'Siamese', 'Sphynx', 'Staffordshire_bull_terrier', 'Wheaten_terrier',
            'Yorkshire_terrier'
        ]
        self.num_classes = len(self.class_names)
        self.history = None
        self.training_time = 0
        
    def create_base_model(self):
        """Base model oluştur"""
        print(f"{self.model_name} base model oluşturuluyor...")
        
        if self.model_name == "EfficientNetB0":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == "MobileNetV2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Desteklenmeyen model: {self.model_name}")
        
        return base_model
    
    def build_model(self, fine_tune=False):
        """Production model oluştur"""
        print("🔧 Production model mimarisi oluşturuluyor...")
        
        base_model = self.create_base_model()
        
        # Fine-tuning için base model'i ayarla
        if fine_tune:
            # Son birkaç katmanı eğitilebilir yap
            base_model.trainable = True
            for layer in base_model.layers[:-10]:
                layer.trainable = False
            print("Fine-tuning aktif")
        else:
            base_model.trainable = False
            print("Base model donduruldu")
        
        # Production-ready model mimarisi
        self.model = tf.keras.Sequential([
            # Base model
            base_model,
            
            # Global pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Batch normalization (daha stabil eğitim)
            tf.keras.layers.BatchNormalization(),
            
            # Dense katmanları
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Çıkış katmanı
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        print(f"Model oluşturuldu: {self.model.count_params():,} parametre")
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Data augmentation ile data generator'lar oluştur"""
        print("Data augmentation ile data generator'lar oluşturuluyor...")
        
        # Training için data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # Validation için sadece normalizasyon
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Data generator'lar
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        print(f"Train: {train_generator.samples} örnek")
        print(f"Validation: {val_generator.samples} örnek")
        
        return train_generator, val_generator
    
    def compile_model(self, learning_rate=0.001):
        """Model derleme"""
        print("Model derleniyor...")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(f" Learning rate: {learning_rate}")
        print("  Optimizer: Adam")
        print("  Loss: Sparse Categorical Crossentropy")
        print("  Metrics: Accuracy, Top-3 Accuracy")
    
    def create_callbacks(self, model_save_path):
        """Training callback'leri oluştur"""
        callbacks = [
            # Early stopping (overfitting'i önle)
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        print("   Callbacks oluşturuldu:")
        print("      - Early Stopping (patience=10)")
        print("      - Reduce LR on Plateau")
        print("      - Model Checkpoint")
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50, fine_tune_epochs=10):
        """İki aşamalı eğitim: Transfer Learning + Fine-tuning"""
        print("\nİki aşamalı eğitim başlıyor...")
        
        # Aşama 1: Transfer Learning
        print("\n AŞAMA 1: Transfer Learning")
        print("=" * 40)
        
        start_time = time.time()
        
        # Callbacks
        callbacks = self.create_callbacks("best_model_stage1.h5")
        
        # İlk eğitim
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Aşama 2: Fine-tuning
        print("\n AŞAMA 2: Fine-tuning")
        print("=" * 40)
        
        # Base model'i eğitilebilir yap
        self.model.layers[0].trainable = True
        
        # Daha düşük learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Fine-tuning eğitimi
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=self.create_callbacks("best_model_final.h5"),
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\n Eğitim tamamlandı!")
        print(f"   - Toplam süre: {self.training_time/60:.2f} dakika")
        print(f"   - Transfer Learning: {len(self.history.history['accuracy'])} epoch")
        print(f"   - Fine-tuning: {len(fine_tune_history.history['accuracy'])} epoch")
    
    def evaluate_model(self, test_generator):
        """Model değerlendirme"""
        print("\n Model değerlendiriliyor...")
        
        # Test performansı
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            test_generator, verbose=0
        )
        
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   Test Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Detaylı analiz
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Sınıf bazında performans
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return {
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy,
            'test_loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def save_model(self, model_path="production_model.h5"):
        """Model ve metadata kaydet"""
        print(f"\n Model kaydediliyor: {model_path}")
        
        # Model kaydet
        self.model.save(model_path)
        
        # Metadata kaydet
        metadata = {
            'model_name': self.model_name,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'training_time': self.training_time,
            'model_params': self.model.count_params()
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("   Model kaydedildi")
        print("   Metadata kaydedildi")
    
    def predict_single_image(self, image_path):
        """Tek resim tahmini"""
        # Resim yükle ve ön işle
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_size, self.img_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        
        # Tahmin
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Top 3 tahmin
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'top3_predictions': [
                {
                    'class': self.class_names[idx],
                    'confidence': predictions[0][idx]
                }
                for idx in top3_indices
            ]
        }
        
        return result

# ===== PRODUCTION PIPELINE =====
def create_production_pipeline():
    """Production pipeline oluştur"""
    print(" PRODUCTION PIPELINE BAŞLATILIYOR")
    print("=" * 60)
    
    # 1. Model oluştur
    classifier = OxfordPetsClassifier(model_name="MobileNetV2")
    classifier.build_model(fine_tune=True)
    
    # 2. Data generator'lar oluştur
    train_dir = "data/train"
    val_dir = "data/val"
    train_generator, val_generator = classifier.create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir
    )
    
    # 3. Model derle
    classifier.compile_model(learning_rate=0.001)
    
    # 4. Eğitim
    classifier.train(train_generator, val_generator, epochs=10, fine_tune_epochs=3)
    
    # 5. Değerlendirme (isteğe bağlı)
    # results = classifier.evaluate_model(val_generator)
    
    # 6. Model kaydet
    classifier.save_model("oxford_pets_production.h5")
    
    print("\n Production pipeline hazır!")
    return classifier

if __name__ == "__main__":
    # Production model oluştur
    classifier = create_production_pipeline()
    
    print("\n PRODUCTION MODEL ÖZELLİKLERİ:")
    print("=" * 40)
    print(" İki aşamalı eğitim (Transfer Learning + Fine-tuning)")
    print(" Data augmentation")
    print(" Early stopping ve learning rate scheduling")
    print(" Batch normalization")
    print(" Dropout regularization")
    print("  Comprehensive evaluation")
    print("  Model checkpointing")
    print("  Metadata saving")
    print("  Single image prediction")
    print("  Production-ready architecture") 