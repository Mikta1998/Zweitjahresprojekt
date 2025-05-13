import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Ensure CPU mode if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
test_dir = 'dataset/split_nested/test'
val_dir = 'dataset/split_nested/val'

# Load model
model = load_model('densenet121/best_densenet_model_7.h5')

# Data generators
val_test_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluation function with PNG saving
def evaluate_and_report(generator, set_name):
    generator.reset()
    y_true = generator.classes
    y_pred_probs = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_names = list(generator.class_indices.keys())

    print(f"\n{set_name} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"{set_name} - Confusion Matrix")
    plt.savefig(f"{set_name.lower().replace(' ', '_')}_confusion_matrix.png", bbox_inches='tight')
    plt.show()

# Run it
evaluate_and_report(val_gen, "Validation Set")
evaluate_and_report(test_gen, "Test Set")