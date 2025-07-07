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

# Import para agrupamento visual 3D e visualização
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from torchvision import models, transforms
import torch
import plotly.graph_objects as go

# ==================== CONFIGURAÇÕES ====================

dataset_path = 'dataset/'
img_size = (128, 128)
batch_size = 16
num_classes = 4
model_save_path = 'modelos/modelo_tuning.h5'
saida_volumes = 'volumes_3d'
os.makedirs(saida_volumes, exist_ok=True)

# ==================== FUNÇÕES PARA AGRUPAMENTO E RECONSTRUÇÃO 3D ====================

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def extrair_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(img_tensor).squeeze().numpy()
    return emb

def reconstruir_volumes_3d():
    for classe in os.listdir(dataset_path):
        classe_path = os.path.join(dataset_path, classe)
        if not os.path.isdir(classe_path):
            continue

        print(f"\n🔍 Processando classe: {classe}")
        arquivos = [f for f in os.listdir(classe_path) if f.lower().endswith(('.png', '.jpg'))]
        caminhos = [os.path.join(classe_path, f) for f in arquivos]

        embeddings = []
        for path in tqdm(caminhos):
            embeddings.append(extrair_embedding(path))

        embeddings = np.array(embeddings)

        cluster_labels = DBSCAN(eps=5.0, min_samples=4).fit_predict(embeddings)
        print(f"🔧 {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} grupos encontrados.")

        agrupados = {}
        for caminho, rotulo in zip(caminhos, cluster_labels):
            if rotulo == -1:
                continue
            agrupados.setdefault(rotulo, []).append(caminho)

        for rotulo, grupo in agrupados.items():
            slices = []
            for caminho_img in sorted(grupo):
                img = Image.open(caminho_img).convert('L').resize(img_size)
                slices.append(np.array(img))
            volume = np.stack(slices, axis=0)
            np.save(os.path.join(saida_volumes, f'{classe}_grupo{rotulo}.npy'), volume)
            print(f"✅ Volume salvo: {classe}_grupo{rotulo}.npy - {volume.shape}")

# ==================== VISUALIZAÇÃO INTERATIVA COM PLOTLY ====================

def visualizar_volume_plotly(volume_path):
    volume = np.load(volume_path)
    num_slices = volume.shape[0]

    fig = go.Figure()
    fig.add_trace(go.Image(z=volume[0]))

    frames = [go.Frame(data=[go.Image(z=volume[k])], name=str(k)) for k in range(num_slices)]
    fig.frames = frames

    sliders = [{
        "currentvalue": {"prefix": "Slice: "},
        "steps": [{
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(k),
            "method": "animate"
        } for k in range(num_slices)]
    }]

    fig.update_layout(
        width=600, height=600,
        sliders=sliders,
        title="Visualização interativa de slices do volume 3D"
    )

    fig.show()

# ==================== FUNÇÕES AUXILIARES PARA RELATÓRIO E GRAD-CAM ====================

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

# ==================== AUGMENTAÇÃO DE DADOS ====================

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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ==================== TREINAMENTO ====================

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

# ==================== RECONSTRUÇÃO 3D E SALVAR VOLUMES ====================

print("\nIniciando agrupamento e reconstrução 3D dos volumes...")
reconstruir_volumes_3d()

# ==================== VISUALIZAÇÃO INTERATIVA DOS VOLUMES ====================

arquivos_volumes = [f for f in os.listdir(saida_volumes) if f.endswith('.npy')]
if arquivos_volumes:
    print(f"\nVisualizando o volume interativo: {arquivos_volumes[0]}")
    visualizar_volume_plotly(os.path.join(saida_volumes, arquivos_volumes[0]))
else:
    print("Nenhum volume 3D encontrado na pasta volumes_3d para visualização.")

# ==================== GRAD-CAM + RELATÓRIO ====================

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
        plt.title(f'{class_name}\nFoco: {foco:.2f}')
        plt.axis('off')
plt.tight_layout()
plt.show()
