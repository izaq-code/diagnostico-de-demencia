# 🧠 NeuroScan: Diagnóstico Inteligente de Demência

> Classificação automatizada de imagens médicas em diferentes **estágios de demência** com apoio de **redes neurais convolucionais** e visualizações com **Grad-CAM**.

---

## 📌 Objetivo

Desenvolver um sistema de **diagnóstico assistido por inteligência artificial (IA)** para classificar imagens de exames cerebrais em **quatro níveis clínicos de demência**, com o objetivo de auxiliar o diagnóstico precoce e monitoramento da progressão da doença.

---

## 🧪 Tecnologias Utilizadas

- **Python 3.11**
- **TensorFlow / Keras**
- **MobileNetV2 (transfer learning)**
- **OpenCV**
- **Scikit-learn**
- **Matplotlib**
- **Grad-CAM** para interpretação dos resultados

---

## 🧠 Classes do Dataset

O modelo é treinado para classificar as imagens em **quatro categorias clínicas**:

1. 🟢 **Não Demente**  
2. 🟡 **Demente Muito Leve**
3. 🟠 **Demente Leve**
4. 🔴 **Demente Moderado**

---

## 📂 Estrutura esperada do diretório `dataset/`

Organize seu conjunto de dados com subpastas nomeadas conforme cada classe:

```
dataset/
├── Nao_Demente/
│   ├── img1.jpg
│   └── ...
├── Demente_Muito_Leve/
│   ├── img1.jpg
│   └── ...
├── Demente_Leve/
│   ├── img1.jpg
│   └── ...
├── Demente_Moderado/
│   ├── img1.jpg
│   └── ...
```

---

## ⚙️ Como Funciona

### 🧹 1. Pré-processamento
- Redimensionamento para `128x128`
- Aumento de dados (data augmentation)
- Normalização para MobileNetV2

### 🧠 2. Arquitetura do Modelo
- **Base:** MobileNetV2 (pré-treinada com ImageNet)
- **Top Layers:** Camadas densas com Dropout e Softmax para 4 classes
- **Treinamento:** Congela a base no início, depois realiza fine-tuning

### 🧪 3. Avaliação e Interpretação
- Relatório de desempenho (precisão, recall, F1)
- Matriz de confusão
- **Grad-CAM**: visualização das regiões da imagem mais relevantes para a decisão do modelo

---

## ▶️ Como Executar

1. Instale os pacotes necessários:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

2. Execute o script principal:

```bash
python lerncode.py
```

---

## 📈 Resultados Esperados

- Acurácia e perda ao longo das épocas
- Classificação das imagens nas 4 categorias
- Grad-CAM para análise visual de decisões da IA

---

## 🔍 Exemplo de saída do Grad-CAM

![Exemplo Grad-CAM](exemplo.jpg)  
*Áreas em vermelho indicam regiões de maior atenção do modelo.*

---

## 👨‍⚕️ Aplicações

- Apoio ao diagnóstico precoce
- Classificação automática em triagens clínicas
- Monitoramento da progressão da demência

---

## 📌 Nome do Projeto

**NeuroScan: Diagnóstico Assistido de Estágios de Demência por IA**