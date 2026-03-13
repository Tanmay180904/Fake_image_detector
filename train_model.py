import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---- Constants ----
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 20      # normal training
EPOCHS_STAGE2 = 8       # fine-tune extra epochs

# ---- Dataset paths (Set 1) ----
train_dir = 'data_new/train'
val_dir   = 'data_new/val'
test_dir  = 'data_new/test'

# ---- Data generators ----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
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

print("Class indices:", train_gen.class_indices)

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

# =====================================================
# STAGE 1: normal training (base frozen)
# =====================================================
base_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.6)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint_stage1 = ModelCheckpoint(
    'model/best_rgb_resnet_stage1.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop_stage1 = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage1 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("==== Stage 1: Training with frozen base ====")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    callbacks=[checkpoint_stage1, early_stop_stage1, reduce_lr_stage1]
)

# =====================================================
# STAGE 2: fine-tune from best of stage 1
# =====================================================
print("==== Stage 2: Fine-tuning top layers ====")

# best stage-1 model se load
model = load_model('model/best_rgb_resnet_stage1.h5')

# last kuch layers ko trainable banao
trainable_count = 40  # need be tweakable
for layer in model.layers[-trainable_count:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

for layer in model.layers[:-trainable_count]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint_stage2 = ModelCheckpoint(
    'model/best_rgb_resnet.h5',   # final best
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop_stage2 = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    callbacks=[checkpoint_stage2, early_stop_stage2, reduce_lr_stage2]
)

# ---- Evaluate on test ----
test_loss, test_acc = model.evaluate(test_gen)
print("Test accuracy:", test_acc)

# ---- Final save for Flask ----
model.save('model/fake_detector_rgb.h5')
print("Final model saved to model/fake_detector_rgb.h5")
