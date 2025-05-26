import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Force CPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
test_dir = 'dataset/split/test'
val_dir = 'dataset/split/val'

# Optional: custom thresholds for specific classes
# Format: {'mel': 0.6, 'nv': 0.4, ...} — set to None to disable
CUSTOM_THRESHOLDS = {
    'akiec': 0.40,
    'bcc':   0.52,
    'bkl':   0.65,
    'df':    0.13,
    'mel':   0.65,
    'nv':    0.97,
    'vasc':  0.07
}
# Load model
model = load_model('best_densenet_model.h5')

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

# Plot probability distributions per class
def plot_all_class_probs(y_pred_probs, class_names):
    for idx, class_name in enumerate(class_names):
        probs = y_pred_probs[:, idx]
        plt.hist(probs, bins=40, alpha=0.75, edgecolor='black')
        plt.title(f'Predicted Probabilities for Class: {class_name}')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Print summary stats per class
def print_prob_stats(y_pred_probs, class_names):
    print("\nClass Probability Summary:")
    for idx, class_name in enumerate(class_names):
        probs = y_pred_probs[:, idx]
        print(f"{class_name:<10} | mean: {probs.mean():.3f} | min: {probs.min():.3f} | max: {probs.max():.3f} | 95%tile: {np.percentile(probs, 95):.3f}")

# Evaluation function
def evaluate_and_report(generator, set_name):
    generator.reset()
    y_true = generator.classes
    y_pred_probs = model.predict(generator, verbose=1)
    class_names = list(generator.class_indices.keys())

    # Apply thresholds if defined and valid
    if CUSTOM_THRESHOLDS:
        if set(CUSTOM_THRESHOLDS.keys()) == set(class_names):
            thresholds = np.array([CUSTOM_THRESHOLDS.get(cls, 0.5) for cls in class_names])
            masked_probs = (y_pred_probs > thresholds) * y_pred_probs
            y_pred = np.argmax(masked_probs, axis=1)

            print("\nApplied thresholds per class:")
            for cls, thresh in zip(class_names, thresholds):
                print(f"{cls:<10}: {thresh}")
        else:
            print("[WARNING] CUSTOM_THRESHOLDS keys do not match detected class names. Skipping thresholding.")
            y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n{set_name} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"{set_name} - Confusion Matrix")
    plt.savefig(f"{set_name.lower().replace(' ', '_')}_confusion_matrix.png", bbox_inches='tight')
    plt.show()

    print_prob_stats(y_pred_probs, class_names)
    plot_all_class_probs(y_pred_probs, class_names)

# Run evaluation
evaluate_and_report(val_gen, "Validation Set")
evaluate_and_report(test_gen, "Test Set")