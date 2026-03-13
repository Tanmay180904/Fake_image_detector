import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = 'data_new/train'
val_dir   = 'data_new/val'
test_dir  = 'data_new/test'

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# yahan jis latest model ko check karna hai uska naam do
model = load_model('model/fake_detector_rgb.h5')  # ya fake_detector_rgb_finetuned.h5
print("Loaded model.")

print("\nEvaluating on TRAIN set...")
train_loss, train_acc = model.evaluate(train_gen)
print("Train accuracy:", train_acc)

print("\nEvaluating on VAL set...")
val_loss, val_acc = model.evaluate(val_gen)
print("Val accuracy:", val_acc)

print("\nEvaluating on TEST set...")
test_loss, test_acc = model.evaluate(test_gen)
print("Test accuracy:", test_acc)
