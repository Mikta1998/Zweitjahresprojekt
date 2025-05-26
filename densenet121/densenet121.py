import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 60

# Paths
train_dir = 'dataset/split/train_balanced'
val_dir = 'dataset/split/val'
test_dir = 'dataset/split/test'

# Data generators (no augmentation)
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)

val_gen = datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

test_gen = datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_densenet_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
callbacks = [early_stop, checkpoint, reduce_lr]

# Model setup with additional Dense layer and regularization
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation='softmax', name='skin_lesion_output')(x)

model = Model(inputs=base_model.input, outputs=output, name="DenseNet121_SkinLesionClassifier")

# Phase 1: Train classifier head
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
history1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=callbacks)

# Phase 2: Fine-tune entire model
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
history2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=callbacks)

# Save final model
model.save('densenet121_skin_cancer_final.h5')

# Merge histories
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return merged

history = merge_histories(history1, history2)

# Plot training history
def plot_history(hist, metric='accuracy'):
    plt.figure(figsize=(10, 4))
    plt.plot(hist[metric], label='Train')
    plt.plot(hist['val_' + metric], label='Val')
    plt.title(f'Model {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history, 'accuracy')
plot_history(history, 'loss')
plot_history(history, 'auc')

# Evaluation function
def evaluate_model(generator, name):
    generator.reset()
    y_true = generator.classes
    y_pred_prob = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    class_names = list(generator.class_indices.keys())

    print(f"\n{name} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# Run evaluation
evaluate_model(val_gen, "Validation Set")
evaluate_model(test_gen, "Test Set")