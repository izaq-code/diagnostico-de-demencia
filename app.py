from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import base64
import io
from PIL import Image
from mistralai import Mistral
import re

app = Flask(__name__)

# Modelo e labels
model = tf.keras.models.load_model('final_model.h5')
last_conv_layer_name = 'Conv_1'
class_labels = ['demente leve', 'demente moderado', 'demente muito leve', 'nao demente']
img_size = (128, 128)

# Cliente Mistral
MISTRAL_API_KEY = "U8PhNbZieOyTvTO3Wf921kV982fGXxBF"
client = Mistral(api_key=MISTRAL_API_KEY)
model_mistral = "mistral-large-latest"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def gerar_explicacao_mistral(class_name, confidence):
    prompt = f"""
Você é um médico especialista em diagnósticos de Alzheimer. Um modelo de Inteligência Artificial analisou uma imagem cerebral e classificou o paciente como {class_name}, com {round(confidence * 100, 2)}% de confiança.

Sua tarefa é obrigatoriamente explicar essa classificação de forma clara, empática e acolhedora.

A explicação deve conter:
- O que significa estar no estágio identificado (leve, moderado, muito leve ou ausência de demência).
- Quais são os sintomas comuns associados a esse estágio.
- Quais ações o paciente e seus familiares devem tomar (consultas médicas, cuidados diários e estilo de vida recomendado).
- A importância da detecção precoce na evolução e no cuidado da doença.

Instruções obrigatórias:
- A classificação ({class_name}) e a porcentagem de confiança ({round(confidence * 100, 2)}%) devem estar claramente mencionadas no início da explicação.
- Não use formatações com asteriscos, como **negrito** pode usar apenas ### titulo ou #### titulo.
- Não utilize frases genéricas como “Estou aqui para ajudar”, “Vamos lá”, “Se tiver dúvidas, me avise” ou “Fico à disposição”.
- Este é um diagnóstico único e definitivo. O paciente e a família não terão oportunidade de enviar perguntas ou continuar esta conversa. Portanto, seja completo e claro em sua explicação.

O tom deve ser direto, humano e acolhedor, como um médico explicando cuidadosamente um diagnóstico importante para o paciente e seus familiares.
"""
    try:
        response = client.chat.complete(
            model=model_mistral,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Erro na API Mistral:", e)
        return "Não foi possível gerar uma explicação no momento. Tente novamente mais tarde."

def formatar_resposta_html(texto_markdown):
    linhas = texto_markdown.split('\n')
    html = ''
    dentro_ul = False
    dentro_ol = False

    for linha in linhas:
        linha = linha.strip()

        if linha.startswith('#### '):
            if dentro_ul:
                html += '</ul>\n'
                dentro_ul = False
            if dentro_ol:
                html += '</ol>\n'
                dentro_ol = False
            subtitulo = linha[5:]
            html += f'<h4>{subtitulo}</h4>\n'

        elif linha.startswith('### '):
            if dentro_ul:
                html += '</ul>\n'
                dentro_ul = False
            if dentro_ol:
                html += '</ol>\n'
                dentro_ol = False
            titulo = linha[4:]
            html += f'<h3>{titulo}</h3>\n'

        elif re.match(r'^\d+\.', linha):  # Lista numerada
            if dentro_ul:
                html += '</ul>\n'
                dentro_ul = False
            if not dentro_ol:
                html += '<ol>\n'
                dentro_ol = True
            item = linha.split('.', 1)[1].strip()
            html += f'  <li>{item}</li>\n'

        elif linha.startswith('- '):  # Lista com marcadores
            if dentro_ol:
                html += '</ol>\n'
                dentro_ol = False
            if not dentro_ul:
                html += '<ul>\n'
                dentro_ul = True
            html += f'  <li>{linha[2:]}</li>\n'

        elif linha == '':
            if dentro_ul:
                html += '</ul>\n'
                dentro_ul = False
            if dentro_ol:
                html += '</ol>\n'
                dentro_ol = False
            html += '<br>\n'

        else:
            # Trata o negrito simples com dois asteriscos
            while '**' in linha:
                linha = linha.replace('**', '<strong>', 1) if linha.count('**') % 2 == 0 else linha.replace('**', '</strong>', 1)
            html += f'<p>{linha}</p>\n'

    if dentro_ul:
        html += '</ul>\n'
    if dentro_ol:
        html += '</ol>\n'

    return html

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Erro ao abrir imagem'}), 400

    img_resized = cv2.resize(img, img_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_preprocessed = img_rgb / 255.0
    img_input = np.expand_dims(img_preprocessed, axis=0).astype(np.float32)

    preds = model.predict(img_input)
    class_idx = np.argmax(preds[0])
    class_name = class_labels[class_idx]
    confidence = float(preds[0][class_idx])

    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name, pred_index=class_idx)
    overlay = overlay_heatmap(img_rgb, heatmap)

    overlay_pil = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="PNG")
    overlay_str = base64.b64encode(buffered.getvalue()).decode()

    explicacao = gerar_explicacao_mistral(class_name, confidence)
    explicacao_html = formatar_resposta_html(explicacao)

    return jsonify({
        'class_name': class_name,
        'confidence': round(confidence, 4),
        'heatmap': 'data:image/png;base64,' + overlay_str,
        'info': explicacao,
        'info_html': explicacao_html
    })

if __name__ == '__main__':
    app.run(debug=True)
