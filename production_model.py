import tensorflow as tf
import os
import json

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_TRANSFER = 10
EPOCHS_FINETUNE = 5

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

MODEL_PATH = "production_model.h5"
METADATA_PATH = "model_metadata.json"

NUM_CLASSES = len(os.listdir(TRAIN_DIR))


def prepare_datasets():
    print("Veri seti hazırlanıyor...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    print(f"{NUM_CLASSES} sınıf bulundu.")
    return train_ds, val_ds


def build_model(fine_tune_at=None):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)]
    )


def get_callbacks(save_path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
    ]


def train(train_ds, val_ds):
    print("Transfer learning başlatılıyor...")
    model = build_model()
    compile_model(model, lr=1e-3)
    callbacks = get_callbacks("best_transfer_model.h5")
    history_transfer = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TRANSFER,
        callbacks=callbacks
    )

    print("Fine-tuning başlatılıyor...")
    model = build_model(fine_tune_at=100)
    compile_model(model, lr=1e-5)
    callbacks = get_callbacks("best_finetune_model.h5")
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks
    )

    return model, history_transfer, history_finetune


def save(model, h1, h2):
    print(f"Model kaydediliyor: {MODEL_PATH}")
    model.save(MODEL_PATH)

    metadata = {
        "transfer_learning_epochs": len(h1.history['accuracy']),
        "fine_tuning_epochs": len(h2.history['accuracy']),
        "final_val_accuracy": round(h2.history['val_accuracy'][-1]*100, 2),
        "model_params": model.count_params(),
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata kaydedildi: {METADATA_PATH}")


def main():
    train_ds, val_ds = prepare_datasets()
    model, h1, h2 = train(train_ds, val_ds)
    save(model, h1, h2)
    print("Eğitim ve kayıt tamamlandı.")


if __name__ == "__main__":
    main()
