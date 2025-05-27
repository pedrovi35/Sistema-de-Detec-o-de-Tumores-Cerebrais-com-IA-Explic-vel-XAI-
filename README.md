

# 🧠 Detecção de Tumores Cerebrais com IA Explicável (XAI) e Assistente Gemini

Este projeto implementa um sistema de deep learning para classificar imagens de Ressonância Magnética (MRI) cerebral, indicando a presença ou ausência de tumores. Além da classificação, o sistema integra técnicas de IA Explicável (XAI) para fornecer insights sobre as decisões do modelo e um assistente de IA (Google Gemini) para interações contextuais.

**Aviso:** Este projeto é destinado a fins educacionais e de demonstração. Não deve ser usado como uma ferramenta de diagnóstico médico. Sempre consulte um profissional de saúde qualificado para diagnósticos e tratamentos.

---

## 📜 Sumário

*   [Visão Geral](#-visão-geral)
*   [✨ Funcionalidades](#-funcionalidades)
*   [🖼️ Demonstração (Interface Streamlit)](#️-demonstração-interface-streamlit)
*   [🛠️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
*   [📂 Estrutura do Projeto](#-estrutura-do-projeto)
*   [⚙️ Configuração e Instalação](#️-configuração-e-instalação)
    *   [Pré-requisitos](#pré-requisitos)
    *   [Obtenção do Dataset](#obtenção-do-dataset)
    *   [Configuração do Ambiente](#configuração-do-ambiente)
    *   [Chave da API Gemini](#chave-da-api-gemini)
*   [🚀 Como Executar](#-como-executar)
    *   [Treinamento do Modelo (Opcional)](#treinamento-do-modelo-opcional)
    *   [Iniciando a Aplicação Streamlit](#iniciando-a-aplicação-streamlit)
*   [🔬 IA Explicável (XAI)](#-ia-explicável-xai)
    *   [Grad-CAM](#grad-cam)
    *   [SHAP](#shap)
    *   [LIME](#lime)
*   [💬 Assistente IA (Gemini)](#-assistente-ia-gemini)
*   [🧠 Detalhes do Modelo](#-detalhes-do-modelo)
*   [📊 Dataset](#-dataset)
*   [🔧 Configurações Globais do Script](#-configurações-globais-do-script)
*   [💡 Possíveis Melhorias Futuras](#-possíveis-melhorias-futuras)
*   [🤝 Contribuições](#-contribuições)
*   [📄 Licença](#-licença)
*   [🙏 Agradecimentos](#-agradecimentos)

---

## 📝 Visão Geral

O objetivo principal é classificar imagens de MRI cerebral como contendo ("yes") ou não ("no") um tumor. Para isso, um modelo de Rede Neural Convolucional (CNN) baseado na arquitetura MobileNetV2 é treinado. A aplicação Streamlit permite que os usuários carreguem suas próprias imagens, recebam uma predição e visualizem explicações XAI para entender melhor a base da decisão do modelo. Adicionalmente, um assistente de IA com o modelo Gemini do Google está integrado para responder perguntas sobre o projeto, o modelo, XAI e informações gerais sobre tumores cerebrais.

---

## ✨ Funcionalidades

*   **Classificação de Imagens:** Prediz se uma imagem de MRI contém um tumor cerebral.
*   **Modelo Pré-treinado (MobileNetV2):** Utiliza transfer learning para maior eficiência e performance.
*   **Treinamento em Duas Fases:**
    1.  Extração de Características (congelando a base MobileNetV2).
    2.  Fine-Tuning (descongelando camadas superiores da MobileNetV2).
*   **Aumento de Dados (Data Augmentation):** Melhora a robustez e generalização do modelo.
*   **Callbacks do Keras:** Inclui `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` e `TensorBoard`.
*   **Avaliação Detalhada:** Geração de relatório de classificação e matriz de confusão.
*   **Interface Web Interativa (Streamlit):**
    *   Upload de imagens de MRI.
    *   Exibição da predição e confiança.
    *   Visualizações de XAI.
*   **Técnicas de IA Explicável (XAI):**
    *   **Grad-CAM:** Destaca regiões importantes na imagem para a decisão.
    *   **SHAP:** Quantifica a contribuição de superpixels/patches para a predição.
    *   **LIME:** Explicações locais baseadas em modelos lineares aproximados em superpixels.
*   **Assistente de IA (Gemini):**
    *   Chat interativo para tirar dúvidas sobre o projeto, o modelo, XAI e tumores cerebrais.
    *   Histórico de conversas.
*   **Cache de Recursos:** `st.cache_resource` para otimizar o carregamento do modelo e explainers.

---

## 🖼️ Demonstração (Interface Streamlit)

*(Adicione aqui screenshots da sua aplicação Streamlit em funcionamento. Por exemplo: a tela de upload, uma predição com Grad-CAM, SHAP, LIME e a interface do chat com Gemini.)*

**Exemplo:**
*   `screenshot_upload.png`: Mostrando a interface de upload.
*   `screenshot_gradcam.png`: Resultado da predição com Grad-CAM.
*   `screenshot_shap.png`: Resultado com SHAP.
*   `screenshot_lime.png`: Resultado com LIME.
*   `screenshot_gemini.png`: Interação com o assistente Gemini.

---

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python 3.x
*   **Deep Learning:**
    *   TensorFlow & Keras (para construção e treinamento do modelo)
    *   MobileNetV2 (arquitetura base)
*   **Processamento de Imagem:**
    *   OpenCV (cv2)
    *   Pillow (PIL)
*   **Análise de Dados e Visualização:**
    *   NumPy
    *   Matplotlib
    *   Seaborn
*   **Interface Web:**
    *   Streamlit
*   **IA Explicável (XAI):**
    *   SHAP
    *   LIME
    *   scikit-image (para segmentação no LIME)
*   **Assistente IA:**
    *   Google Generative AI (SDK para Gemini)
    *   python-dotenv (para gerenciamento de chaves de API)
*   **Métricas:**
    *   scikit-learn (para `classification_report`, `confusion_matrix`)

---

## 📂 Estrutura do Projeto

---

## ⚙️ Configuração e Instalação

### Pré-requisitos

*   Python (versão 3.8 ou superior recomendada)
*   `pip` (gerenciador de pacotes Python)
*   Opcional: Um ambiente virtual (venv, conda) é altamente recomendado.

### Obtenção do Dataset

1.  O dataset utilizado neste projeto é o "Brain Tumor Classification (MRI)" disponível no Kaggle: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
2.  Baixe o dataset e extraia-o.
3.  Crie um diretório chamado `brain_tumor_dataset` na raiz do seu projeto.
4.  Dentro de `brain_tumor_dataset`, mova as imagens para as subpastas `yes` (para imagens com tumor) e `no` (para imagens sem tumor), conforme a estrutura original do dataset.

    ```
    brain_tumor_dataset/
    ├── yes/
    │   ├── Y1.jpg
    │   ├── Y2.jpg
    │   └── ...
    └── no/
        ├── N1.jpg
        ├── N2.jpg
        └── ...
    ```

### Configuração do Ambiente

1.  **Clone o repositório (se aplicável):**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DO_REPOSITORIO>
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo (ou gere-o a partir do seu ambiente se já tiver tudo instalado):
    ```txt
    tensorflow>=2.8.0
    opencv-python>=4.5.0
    numpy>=1.20.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    Pillow>=8.4.0
    streamlit>=1.10.0
    shap>=0.40.0
    lime>=0.2.0
    scikit-image>=0.19.0
    scikit-learn>=1.0.0
    google-generativeai>=0.3.0
    python-dotenv>=0.20.0
    ```
    Então, instale-as:
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Dependendo do seu sistema e se você tem uma GPU NVIDIA, pode ser necessário instalar `tensorflow-gpu` em vez de `tensorflow` e configurar CUDA/cuDNN.*

### Chave da API Gemini

1.  Para usar o assistente Gemini, você precisará de uma chave de API do Google AI Studio.
    *   Visite [https://aistudio.google.com/](https://aistudio.google.com/) e obtenha sua chave.
2.  Crie um arquivo chamado `.env` na raiz do seu projeto.
3.  Adicione sua chave de API ao arquivo `.env` da seguinte forma:
    ```
    GOOGLE_API_KEY="SUA_CHAVE_API_AQUI"
    ```
    **Importante:** Adicione `.env` ao seu arquivo `.gitignore` para não versionar sua chave de API.

---

## 🚀 Como Executar

### Treinamento do Modelo (Opcional)

O script é configurado para treinar o modelo automaticamente se o arquivo `brain_tumor_cnn_model_mobilenetv2_unified_best.h5` não for encontrado na raiz do projeto, ou se a variável `FORCE_TRAIN_MODEL` no script estiver definida como `True`.

*   Para forçar o treinamento, modifique a variável `FORCE_TRAIN_MODEL = True` no início do script `your_script_name.py`.
*   Se o modelo já existe e `FORCE_TRAIN_MODEL = False`, o script carregará o modelo existente.

O treinamento pode ser demorado. Os logs de treinamento (incluindo para TensorBoard) são salvos no diretório `logs/fit/`. Você pode monitorar o treinamento com TensorBoard:
```bash
tensorboard --logdir logs/fit

