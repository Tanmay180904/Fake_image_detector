import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dropout

# ---- Constants ----
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FT = 6   # thoda kam epochs, time bachega

# ---- Dataset paths (Set 1) ----
train_dir = 'data_new/train'
val_dir   = 'data_new/val'
test_dir  = 'data_new/test'

# ---- Data generators ----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

os.makedirs('model', exist_ok=True)

# ----- yahan tumhara latest best model (data_new wala) -----
best_model_path = 'model/fake_detector_rgb.h5'
# -----------------------------------------------------------

model = load_model(best_model_path)
print("Loaded best model from:", best_model_path)

# ---- OPTIONAL: head me dropout badhana (agar already 0.6 hai to skip) ----
# yeh part sirf tab chalega jab last se third layer Dropout ho
try:
    if isinstance(model.layers[-3], Dropout):
        model.layers[-3].rate = 0.7   # 0.6 -> 0.7
        print("Increased Dropout rate to 0.7 in head.")
except Exception:
    pass

# ---- last kam layers ko trainable banao (40 -> 20) ----
trainable_count = 20
for layer in model.layers[-trainable_count:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

for layer in model.layers[:-trainable_count]:
    layer.trainable = False

# ---- Compile with lower LR for fine-tuning ----
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint_ft = ModelCheckpoint(
    'model/fake_detector_rgb_finetuned.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop_ft = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_ft = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("==== Fine-tuning from existing best model (20 layers, higher dropout) ====")
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FT,
    callbacks=[checkpoint_ft, early_stop_ft, reduce_lr_ft]
)

# ---- Test evaluation after fine-tune ----
test_loss, test_acc = model.evaluate(test_gen)
print("Test accuracy after fine-tune:", test_acc)

print("Fine-tuning finished. Best model saved to model/fake_detector_rgb_finetuned.h5")
