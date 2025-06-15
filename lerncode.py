import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
dataset_path = 'dataset/'
img_size = (128, 128)
batch_size = 16
num_classes = 4
epochs = 15
model_save_path = 'best_model.h5'

# Geradores de dados
datagen = ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    preprocessing_function=preprocess_input
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Modelo com MobileNetV2
input_tensor = Input(shape=(img_size[0], img_size[1], 3))
base_model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Treinar com camadas congeladas
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

# Fine-tuning
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# Avaliação final
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print("Relatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred))

# Salvar modelo final no novo formato
model.save('final_model.h5')

# ==============================
# GRAD-CAM VISUALIZAÇÃO
# ==============================

def get_img_by_class(class_name, img_size):
    class_path = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(class_path, fname)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            return preprocess_input(img_array), img_path
    return None, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Corrigido para evitar warning

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-6)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# Exibir uma imagem por classe com Grad-CAM
selected_imgs = []
for class_name in class_labels:
    img_array, img_path = get_img_by_class(class_name, img_size)
    if img_array is not None:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
        result_img = display_gradcam(img_path, heatmap)
        selected_imgs.append((class_name, result_img))

# Plotar imagens com Grad-CAM
plt.figure(figsize=(16, 6))
for i, (label, img) in enumerate(selected_imgs):
    plt.subplot(1, len(selected_imgs), i + 1)
    plt.imshow(img)
    plt.title(f'Classe: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# ==============================
# GRÁFICO DE ACURÁCIA (EXTRA)
# ==============================
def plot_training(history, title='Treinamento'):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title(f'{title} - Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training(history, title='Fase 1')
plot_training(history_fine, title='Fine-Tuning')
