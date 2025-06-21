<<<<<<< HEAD
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==================== PRE-PROCESSAMENTO CLÍNICO ====================

def cortar_apenas_cerebro(imagem):
    """Remove margens e áreas externas à região encefálica."""
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return imagem
    x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    return imagem[y:y+h, x:x+w]

def equalizar_contraste(img):
    """Equaliza contraste para realçar detalhes internos."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)

def preprocessar_img_cv(img_path, target_size):
    """Carrega, recorta, equaliza e redimensiona imagem."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cortar_apenas_cerebro(img)
    img = equalizar_contraste(img)
    img = cv2.resize(img, target_size)
    return img

# ==================== CONFIGURAÇÕES ====================

dataset_path = 'dataset/'
img_size = (128, 128)
batch_size = 16
num_classes = 4
model_save_path = 'modelo_clinico_melhorado.h5'

# ==================== GERADORES ====================

datagen = ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
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

# ==================== MODELO ====================

input_tensor = Input(shape=(img_size[0], img_size[1], 3))
base_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=callbacks)

# Fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=callbacks)

# ==================== GRAD-CAM + RELATÓRIO ====================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv', pred_index=None):
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
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

def gerar_mascara_hipocampo_simulada(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cv2.rectangle(mask, (int(w*0.35), int(h*0.4)), (int(w*0.65), int(h*0.6)), 1, -1)
    return mask

def calcular_foco_em_regiao(heatmap, mask):
    heatmap_resized = cv2.resize(heatmap, mask.shape[::-1])
    return np.sum(heatmap_resized * mask) / np.sum(mask)

def gerar_relatorio(img_path, class_name, class_labels, model):
    img = preprocessar_img_cv(img_path, img_size)
    img_input = preprocess_input(img.copy().astype(np.float32))
    heatmap = make_gradcam_heatmap(img_input, model)
    superimposed = display_gradcam(img, heatmap)
    foco = calcular_foco_em_regiao(heatmap, gerar_mascara_hipocampo_simulada(img_size))

    # Salvar imagem e gerar relatório
    caminho_img = f'gradcam_{class_name}.png'
    cv2.imwrite(caminho_img, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Diagnóstico Assistido por IA", ln=True)
    pdf.cell(200, 10, txt=f"Classe Prevista: {class_name}", ln=True)
    pdf.cell(200, 10, txt=f"Foco no Hipocampo (Simulado): {foco:.2f}", ln=True)
    pdf.image(caminho_img, x=10, y=40, w=120)
    pdf.output(f'relatorio_{class_name}.pdf')

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

# Executar relatório para uma imagem por classe
class_labels = list(train_generator.class_indices.keys())
for class_name in class_labels:
    class_path = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('jpg', 'png')):
            img_path = os.path.join(class_path, fname)
            gerar_relatorio(img_path, class_name, class_labels, model)
            break
=======
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
dataset_path = 'dataset/'
img_size = (128, 128)
batch_size = 16
num_classes = 4
model_save_path = 'modelo_tuning.h5'

# Funções auxiliares
def gerar_mascara_simulada(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cv2.rectangle(mask, (int(w*0.35), int(h*0.35)), (int(w*0.65), int(h*0.65)), 1, -1)
    return mask

def calcular_foco_em_regiao(heatmap, mask):
    heatmap_resized = cv2.resize(heatmap, mask.shape[::-1])
    return np.sum(heatmap_resized * mask) / np.sum(mask)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
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

def salvar_gradcam_imagem(img, caminho="gradcam_output.png"):
    cv2.imwrite(caminho, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def gerar_relatorio_pdf(predicao, foco, caminho_img, nome_arquivo="relatorio_diagnostico.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Diagnóstico Assistido por IA", ln=True)
    pdf.cell(200, 10, txt=f"Classe Prevista: {predicao}", ln=True)
    pdf.cell(200, 10, txt=f"Foco no Hipocampo (Simulado): {foco:.2f}", ln=True)
    pdf.image(caminho_img, x=10, y=40, w=120)
    pdf.output(nome_arquivo)

def get_img_by_class(class_name, img_size):
    class_path = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(class_path, fname)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            return preprocess_input(img_array), img_path
    return None, None

# Augmentação de dados aprimorada
datagen = ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
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

# Modelo com EfficientNetB0
input_tensor = Input(shape=(img_size[0], img_size[1], 3))
base_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Congela camadas base
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Treinamento inicial
model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[checkpoint, early_stop, reduce_lr])

# Fine-tuning: descongelar últimas camadas
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[checkpoint, early_stop, reduce_lr])

# Grad-CAM + Relatório
class_labels = list(train_generator.class_indices.keys())
plt.figure(figsize=(16, 6))
for i, class_name in enumerate(class_labels):
    img_array, img_path = get_img_by_class(class_name, img_size)
    if img_array is not None:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv')
        result_img = display_gradcam(img_path, heatmap)
        mask = gerar_mascara_simulada(img_size)
        foco = calcular_foco_em_regiao(heatmap, mask)
        salvar_gradcam_imagem(result_img, f"gradcam_{class_name}.png")
        gerar_relatorio_pdf(class_name, foco, f"gradcam_{class_name}.png", f"relatorio_{class_name}.pdf")
        plt.subplot(1, len(class_labels), i + 1)
        plt.imshow(result_img)
        plt.title(f'{class_name}\\nFoco: {foco:.2f}')
        plt.axis('off')
plt.tight_layout()
plt.show()
>>>>>>> 589b37f4e9f9fe943538671c2da8e9a830005b09
