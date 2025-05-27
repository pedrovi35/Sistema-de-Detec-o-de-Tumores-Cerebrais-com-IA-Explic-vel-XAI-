# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import datetime
import shutil

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# XAI Libraries
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    import shap
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    STREAMLIT_AVAILABLE = True
except ImportError:
    print(
        "INFO: Streamlit ou bibliotecas XAI (shap, lime) não encontradas. A interface Streamlit não estará disponível.")

# Gemini AI Assistant Libraries
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GEMINI_AVAILABLE = True
except ImportError:
    print("INFO: google-generativeai ou python-dotenv não encontradas. O assistente Gemini não estará disponível.")

# --- Configurações Globais ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 16
EPOCHS_FEATURE_EXTRACTION = 25
EPOCHS_FINE_TUNING = 35
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

DATA_DIR = 'brain_tumor_dataset'
MODEL_SAVE_PATH = 'brain_tumor_cnn_model_mobilenetv2_unified_best.h5'
LOG_DIR_BASE = "logs/fit/"
LOG_DIR = LOG_DIR_BASE + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

FORCE_TRAIN_MODEL = False

# --- Gemini Configuration ---
GOOGLE_API_KEY = None
if GEMINI_AVAILABLE:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
        except Exception as e:
            print(f"Erro ao configurar Gemini API: {e}. O assistente Gemini pode não funcionar.")
    else:
        if STREAMLIT_AVAILABLE and hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit:
            pass
        else:
            print("INFO: Chave GOOGLE_API_KEY não encontrada no ambiente. O assistente Gemini não estará disponível.")


# --- 1. Funções de Treinamento e Modelo ---
def build_model_with_mobilenetv2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, name='mobilenetv2_base')
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(256, activation='relu', name="fc1")(x)
    x = Dropout(0.5, name="dropout1")(x)
    outputs = Dense(1, activation='sigmoid', name="output_sigmoid")(x)
    model = Model(inputs, outputs, name="BrainTumorMobileNetV2")
    return model


def train_and_save_model():
    global LOG_DIR
    LOG_DIR = LOG_DIR_BASE + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(DATA_DIR) or \
            not os.path.exists(os.path.join(DATA_DIR, 'yes')) or \
            not os.path.exists(os.path.join(DATA_DIR, 'no')):
        msg = f"ERRO: Diretório de dados '{DATA_DIR}' ou subdiretórios 'yes'/'no' não encontrados."
        print(msg)
        if STREAMLIT_AVAILABLE and hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit:
            st.error(msg + " Verifique a configuração.")
        return None

    print("Iniciando o processo de treinamento do modelo (MobileNetV2)...")
    os.makedirs(LOG_DIR, exist_ok=True)

    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_v2_preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest',
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42
        )
        validation_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42
        )
    except Exception as e:
        msg = f"Erro ao criar geradores de dados: {e}"
        print(msg)
        if STREAMLIT_AVAILABLE and hasattr(st,
                                           '_is_running_with_streamlit') and st._is_running_with_streamlit: st.error(
            msg)
        return None

    if train_generator.n == 0 or validation_generator.n == 0:
        msg = "ERRO: Nenhum dado encontrado nos geradores de treino ou validação."
        print(msg)
        if STREAMLIT_AVAILABLE and hasattr(st,
                                           '_is_running_with_streamlit') and st._is_running_with_streamlit: st.error(
            msg)
        return None

    print(f"Encontradas {train_generator.n} imagens para treino, {validation_generator.n} para validação.")
    print(f"Classes: {train_generator.class_indices}")
    target_names = list(train_generator.class_indices.keys())

    model = build_model_with_mobilenetv2()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary(
        print_fn=lambda x: print(x) if not (STREAMLIT_AVAILABLE and hasattr(st,
                                                                            '_is_running_with_streamlit') and st._is_running_with_streamlit) else st.text(
            x))
    print("\n--- Fase 1: Treinando a cabeça (MobileNetV2 congelado) ---")

    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=LOG_DIR + "/phase1", histogram_freq=1)
    ]

    history_phase1 = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=EPOCHS_FEATURE_EXTRACTION,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        callbacks=callbacks_phase1
    )

    print("\n--- Fase 2: Fine-tuning (descongelando camadas do MobileNetV2) ---")
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(MODEL_SAVE_PATH)
        print("Modelo carregado do checkpoint para fine-tuning.")
    else:
        print("AVISO: Nenhum modelo salvo da Fase 1 encontrado. Continuando com o modelo atual em memória.")

    try:
        base_model_loaded = model.get_layer(name='mobilenetv2_base')
    except ValueError:
        msg = "ERRO: Camada base 'mobilenetv2_base' não encontrada no modelo carregado para fine-tuning."
        print(msg)
        if STREAMLIT_AVAILABLE and hasattr(st,
                                           '_is_running_with_streamlit') and st._is_running_with_streamlit: st.error(
            msg)
        return None

    base_model_loaded.trainable = True
    fine_tune_layers_count = 30
    for i, layer in enumerate(base_model_loaded.layers):
        if i < (len(base_model_loaded.layers) - fine_tune_layers_count):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary(
        print_fn=lambda x: print(x) if not (STREAMLIT_AVAILABLE and hasattr(st,
                                                                            '_is_running_with_streamlit') and st._is_running_with_streamlit) else st.text(
            x))

    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=LOG_DIR + "/phase2", histogram_freq=1)
    ]

    initial_epoch_phase2 = 0
    if hasattr(history_phase1, 'epoch') and history_phase1.epoch:
        initial_epoch_phase2 = history_phase1.epoch[-1] + 1

    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=EPOCHS_FEATURE_EXTRACTION + EPOCHS_FINE_TUNING,
        initial_epoch=initial_epoch_phase2,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        callbacks=callbacks_phase2
    )

    print("\n--- Avaliando o MELHOR modelo treinado (do ModelCheckpoint) no conjunto de validação ---")
    best_model_final = load_model(MODEL_SAVE_PATH)

    validation_generator.reset()
    loss, acc, prec, rec = best_model_final.evaluate(validation_generator,
                                                     steps=max(1, validation_generator.samples // BATCH_SIZE),
                                                     verbose=1)
    print(f"Validação Final - Perda: {loss:.4f}, Acurácia: {acc:.4f}, Precisão: {prec:.4f}, Recall: {rec:.4f}")

    validation_generator.reset()
    y_pred_probs = best_model_final.predict(validation_generator,
                                            steps=max(1, validation_generator.samples // BATCH_SIZE))
    y_pred_probs = y_pred_probs[:validation_generator.samples]
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = validation_generator.classes

    print("\nRelatório de Classificação Final (Validação):")
    print(classification_report(y_true, y_pred_classes, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão (Validação)')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    if not (STREAMLIT_AVAILABLE and hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit):
        plt.show()
    else:
        cm_path = "confusion_matrix_validation.png"
        plt.savefig(cm_path)
        plt.close()
        if 'cm_image_path' not in st.session_state:
            st.session_state.cm_image_path = cm_path

    print(f"Melhor modelo salvo em: {MODEL_SAVE_PATH}")
    print("Treinamento concluído.")
    return best_model_final


# --- 2. Funções da Aplicação Streamlit (e XAI) ---
if STREAMLIT_AVAILABLE:
    @st.cache_resource
    def load_streamlit_model_cached(_model_path):
        try:
            model_loaded = tf.keras.models.load_model(_model_path)
            last_conv_layer_name_for_gradcam = None
            base_model_layer_name_for_gradcam = None

            # DEBUG: Mostrar nome do modelo carregado
            if hasattr(st, 'write'): st.write(f"DEBUG: Modelo carregado: {model_loaded.name}")

            try:
                base_model_for_gradcam = model_loaded.get_layer(name='mobilenetv2_base')
                base_model_layer_name_for_gradcam = base_model_for_gradcam.name
                if hasattr(st, 'write'): st.write(
                    f"DEBUG: Camada base para Grad-CAM encontrada: {base_model_layer_name_for_gradcam}")

                # Lógica para encontrar a última camada CONV ou ReLU dentro da base
                # Preferência por 'out_relu' se existir, ou a última Conv/BN/ReLU significativa
                for layer in reversed(base_model_for_gradcam.layers):
                    if layer.name == 'out_relu':  # Camada comum em MobileNetV2 antes de GAP
                        last_conv_layer_name_for_gradcam = layer.name
                        break
                    # Considerar camadas de convolução ou as que seguem (BN, ReLU) como candidatas
                    # A última camada antes do GAP (que está fora da base_model aqui)
                    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D,
                                          tf.keras.layers.BatchNormalization, tf.keras.layers.ReLU)):
                        if not isinstance(layer, (tf.keras.layers.InputLayer,
                                                  tf.keras.layers.ZeroPadding2D)):  # Evitar camadas não representativas
                            last_conv_layer_name_for_gradcam = layer.name
                            break  # Pega a primeira encontrada na ordem reversa (última do modelo)

                if not last_conv_layer_name_for_gradcam and base_model_for_gradcam.layers:  # Fallback se nada específico foi encontrado
                    last_conv_layer_name_for_gradcam = base_model_for_gradcam.layers[
                        -1].name  # Pega a última camada da base
                    if hasattr(st, 'write'): st.write(
                        f"DEBUG: Fallback dentro da base: Usando última camada '{last_conv_layer_name_for_gradcam}' de '{base_model_layer_name_for_gradcam}'")

            except ValueError:
                if hasattr(st, 'warning'): st.warning(
                    "DEBUG: Não foi possível encontrar a camada base 'mobilenetv2_base'. Tentando fallback geral no modelo principal.")
                for layer in reversed(model_loaded.layers):
                    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        last_conv_layer_name_for_gradcam = layer.name
                        base_model_layer_name_for_gradcam = None  # Indicar que não está na base nomeada
                        if hasattr(st, 'write'): st.write(
                            f"DEBUG: Fallback geral: Usando camada convolucional do modelo principal: {last_conv_layer_name_for_gradcam}")
                        break

            if not last_conv_layer_name_for_gradcam:
                if hasattr(st, 'error'): st.error(
                    "Grad-CAM ERRO: Não foi possível determinar a camada convolucional final. Grad-CAM pode não funcionar.")
                last_conv_layer_name_for_gradcam = "conv2d_fallback_placeholder"  # Placeholder para evitar erros
            else:
                if hasattr(st, 'success'): st.success(
                    f"Grad-CAM INFO: Usará a camada: '{last_conv_layer_name_for_gradcam}' (encontrada em '{base_model_layer_name_for_gradcam if base_model_layer_name_for_gradcam else 'modelo principal'}')")

            return model_loaded, last_conv_layer_name_for_gradcam, base_model_layer_name_for_gradcam

        except Exception as e:
            if hasattr(st, 'error'): st.error(f"Erro Crítico ao carregar o modelo para Streamlit: {e}")
            if hasattr(st, 'exception'): st.exception(e)  # Mostra o traceback completo no Streamlit
            return None, None, None


    def preprocess_image_for_streamlit(image_pil, target_size=(IMG_WIDTH, IMG_HEIGHT)):
        img = image_pil.convert('RGB').resize(target_size)
        img_array_uint8 = np.array(img)
        img_array_float32 = img_array_uint8.astype(np.float32)
        img_preprocessed = mobilenet_v2_preprocess_input(np.expand_dims(img_array_float32.copy(), axis=0))
        return img_array_uint8, img_preprocessed


    def make_gradcam_heatmap_st(img_array_processed, model_st_obj, last_conv_name_st, base_model_name_for_gradcam_st):
        try:
            model_inputs = model_st_obj.inputs

            if base_model_name_for_gradcam_st:
                base_model_layer = model_st_obj.get_layer(name=base_model_name_for_gradcam_st)
                conv_layer = base_model_layer.get_layer(name=last_conv_name_st)
                conv_output = conv_layer.output
            else:
                conv_layer = model_st_obj.get_layer(name=last_conv_name_st)
                conv_output = conv_layer.output

            classifier_output = model_st_obj.output
            grad_model = Model(inputs=model_inputs, outputs=[conv_output, classifier_output])

        except ValueError as e:
            st.error(f"GradCAM: Erro ao construir grad_model. Camada '{last_conv_name_st}'"
                     f"{' em ' + base_model_name_for_gradcam_st if base_model_name_for_gradcam_st else ''} não encontrada ou inválida. Erro: {e}")
            try:
                # Tenta obter as dimensões esperadas do heatmap
                temp_shape_info = model_st_obj.get_layer(name=base_model_name_for_gradcam_st).get_layer(
                    name=last_conv_name_st).output_shape if base_model_name_for_gradcam_st else model_st_obj.get_layer(
                    name=last_conv_name_st).output_shape
                return np.zeros((temp_shape_info[1], temp_shape_info[2]))
            except:
                return np.zeros((IMG_HEIGHT // 32, IMG_WIDTH // 32))

        with tf.GradientTape() as tape:
            # tape.watch(img_array_processed) # Geralmente não necessário para inputs de modelo
            conv_outputs_value, preds = grad_model(img_array_processed, training=False)  # Adicionado training=False

            if preds.shape[-1] == 1:  # Saída sigmoide
                class_channel = preds[:, 0]
            else:  # Saída softmax (improvável para este modelo, mas para robustez)
                # Explicar a classe com maior probabilidade
                predicted_class_index = tf.argmax(preds[0])
                class_channel = preds[:, predicted_class_index]

        grads = tape.gradient(class_channel, conv_outputs_value)
        if grads is None:
            st.error(
                "GradCAM: Gradientes são None. A camada pode não ser diferenciável ou não está no grafo de gradiente. "
                f"Verifique se '{last_conv_name_st}' é uma camada convolucional ou uma camada de ativação/BN que a segue diretamente."
            )
            # Retorna um heatmap vazio com as dimensões esperadas da camada convolucional
            return np.zeros((conv_outputs_value.shape[1], conv_outputs_value.shape[2]))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_value_no_batch = conv_outputs_value[0]
        heatmap = conv_outputs_value_no_batch @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()


    def display_gradcam_st(original_img_uint8, heatmap, alpha=0.6):
        if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
            st.warning("GradCAM heatmap está vazio ou zerado. Não será sobreposto.")
            return original_img_uint8
        heatmap_resized = cv2.resize(heatmap, (original_img_uint8.shape[1], original_img_uint8.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed_img


    @st.cache_resource
    def get_shap_explainer_st(_model, _background_data):
        try:
            return shap.GradientExplainer(_model, _background_data)
        except Exception as e:
            st.warning(f"Falha ao criar SHAP GradientExplainer ({e}). SHAP pode não funcionar.")
            return None


    @st.cache_resource
    def get_lime_explainer_st():
        return lime_image.LimeImageExplainer(random_state=42)


    def initialize_gemini_chat():
        if not GEMINI_AVAILABLE or not GOOGLE_API_KEY:
            if not GOOGLE_API_KEY and GEMINI_AVAILABLE:
                if hasattr(st, 'sidebar') and hasattr(st.sidebar, 'warning'): st.sidebar.warning(
                    "Assistente AI: Chave GOOGLE_API_KEY não configurada.")
            return None
        try:
            system_instruction = (
                "Você é um assistente de IA especializado em análise de imagens de ressonância magnética (MRI) cerebral "
                "para detecção de tumores. Você está integrado a uma aplicação Streamlit que usa um modelo de deep learning "
                "baseado em MobileNetV2 para classificar imagens como 'Com Tumor' ou 'Sem Tumor'. "
                "Suas respostas devem focar em: \n"
                "1. Interpretação geral de MRIs cerebrais no contexto de tumores (sem dar diagnóstico médico direto, sempre recomende um especialista).\n"
                "2. Explicação do funcionamento do modelo MobileNetV2 de forma simplificada.\n"
                "3. Como as técnicas de XAI (Grad-CAM, SHAP, LIME) ajudam a entender as decisões do modelo.\n"
                "4. Informações gerais sobre tumores cerebrais (sintomas comuns, tipos, importância da detecção precoce), sempre recomendando consulta médica.\n"
                "5. Funcionalidades da aplicação Streamlit onde você está inserido.\n"
                "Evite tópicos não relacionados a este domínio. Seja conciso, informativo e útil. "
                "Se perguntarem algo que você não sabe ou que seria um diagnóstico, recomende consultar um especialista médico."
            )
            model_gemini = genai.GenerativeModel(
                model_name='gemini-1.5-flash-latest',
                system_instruction=system_instruction
            )
            return model_gemini.start_chat(history=[])
        except Exception as e:
            if hasattr(st, 'sidebar') and hasattr(st.sidebar, 'error'): st.sidebar.error(
                f"Erro ao inicializar o chat com Gemini: {e}")
            return None


    def run_streamlit_app_interface(model_path_for_st=MODEL_SAVE_PATH):
        st.set_page_config(layout="wide", page_title="Detecção de Tumores Cerebrais XAI")
        st.title("🔬 Sistema de Detecção de Tumores Cerebrais com IA Explicável (XAI)")
        st.markdown("""
        Esta aplicação utiliza um modelo de Deep Learning (baseado em **MobileNetV2**) para classificar imagens de Ressonância Magnética (MRI)
        cerebral. Além da predição, são fornecidas explicações visuais para entender as decisões do modelo.
        **AVISO: Esta ferramenta é para fins educacionais e de demonstração. Não substitui o diagnóstico médico profissional.**
        """)

        st.sidebar.header("Controles")
        uploaded_file = st.sidebar.file_uploader("Carregue uma imagem de MRI (.jpg, .jpeg, .png)",
                                                 type=["jpg", "jpeg", "png"])

        st.sidebar.markdown("---")
        st.sidebar.subheader("Sobre o Modelo:")
        class_names_st = {0: 'Sem Tumor', 1: 'Com Tumor'}
        st.sidebar.info(f"""
        - **Arquitetura Base:** MobileNetV2 
        - **Classes:** {', '.join(class_names_st.values())}
        - **Input:** {IMG_WIDTH}x{IMG_HEIGHT} pixels
        """)
        if os.path.exists(model_path_for_st):
            model_size_mb = os.path.getsize(model_path_for_st) / (1024 * 1024)
            st.sidebar.caption(f"Modelo: `{os.path.basename(model_path_for_st)}` ({model_size_mb:.2f} MB)")
        else:
            st.sidebar.caption(f"Modelo: (Ainda não treinado/salvo)")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Técnicas XAI:")
        st.sidebar.markdown("- **Grad-CAM:** Relevância de regiões da imagem.")
        st.sidebar.markdown("- **SHAP:** Importância de patches/superpixels.")
        st.sidebar.markdown("- **LIME:** Explicação local com superpixels.")

        if GEMINI_AVAILABLE:
            st.sidebar.markdown("---")
            with st.sidebar.expander("💬 Assistente AI (Gemini)", expanded=False):
                if "gemini_chat" not in st.session_state:
                    st.session_state.gemini_chat = initialize_gemini_chat()
                if st.session_state.gemini_chat:
                    for message in st.session_state.gemini_chat.history:
                        role = "user" if message.role == "user" else "assistant"
                        with st.chat_message(role):
                            st.markdown(message.parts[0].text)
                    if prompt := st.chat_input("Pergunte sobre a aplicação, o modelo, XAI ou tumores cerebrais..."):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        try:
                            with st.spinner("Gemini está pensando..."):
                                response = st.session_state.gemini_chat.send_message(prompt)
                            with st.chat_message("assistant"):
                                st.markdown(response.text)
                        except Exception as e_gemini_send:
                            st.error(f"Erro ao comunicar com Gemini: {e_gemini_send}")
                elif not GOOGLE_API_KEY:
                    st.warning("Assistente AI: Chave GOOGLE_API_KEY não configurada.")
                else:
                    st.warning("Não foi possível inicializar o assistente AI. Verifique os logs.")
        else:
            st.sidebar.markdown("---")
            st.sidebar.info("Assistente Gemini não disponível (bibliotecas ausentes).")

        model_st_obj, last_conv_name_st, base_model_name_st_gradcam = load_streamlit_model_cached(model_path_for_st)

        if model_st_obj is None:
            st.error(
                f"O modelo treinado em '{model_path_for_st}' não pôde ser carregado. Verifique o caminho ou se o treinamento foi concluído com sucesso.")
            if st.button("Tentar Treinar Modelo Agora"):
                with st.spinner(
                        "Treinamento em progresso... Isso pode levar vários minutos. A página será recarregada ao concluir."):
                    train_and_save_model()
                    st.rerun()
            return

        if 'shap_background' not in st.session_state:
            num_bg_samples = min(5, BATCH_SIZE)
            dummy_bg_st = np.random.rand(num_bg_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32) * 255.0
            st.session_state.shap_background = mobilenet_v2_preprocess_input(dummy_bg_st.copy())

        shap_explainer_st_obj = get_shap_explainer_st(model_st_obj, st.session_state.shap_background)
        lime_explainer_st_obj = get_lime_explainer_st()

        if 'cm_image_path' in st.session_state and os.path.exists(st.session_state.cm_image_path):
            st.sidebar.markdown("---")
            st.sidebar.subheader("Performance do Modelo (Validação)")
            try:
                st.sidebar.image(st.session_state.cm_image_path, caption="Matriz de Confusão (Treinamento)")
            except Exception as e_cm_img:
                st.sidebar.warning(f"Não foi possível exibir a matriz de confusão: {e_cm_img}")

        if uploaded_file is not None:
            try:
                image_pil_st = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Erro ao abrir a imagem: {e}")
                return

            original_img_uint8_st, processed_img_for_model_st = preprocess_image_for_streamlit(image_pil_st)
            col_img, col_pred_xai = st.columns([0.4, 0.6])

            with col_img:
                st.subheader("Imagem Analisada")
                st.image(original_img_uint8_st, caption="Imagem Original (Redimensionada)", use_column_width=True)
                with st.spinner("Analisando imagem..."):
                    prediction_prob_st = model_st_obj.predict(processed_img_for_model_st)[0][0]
                predicted_class_idx_st = 1 if prediction_prob_st > 0.5 else 0
                predicted_class_name_st = class_names_st[predicted_class_idx_st]
                st.subheader("Resultado da Predição")
                confidence_score = prediction_prob_st if predicted_class_idx_st == 1 else 1 - prediction_prob_st
                if predicted_class_idx_st == 1:
                    st.error(f"**{predicted_class_name_st}** (Confiança: {confidence_score:.2%})")
                else:
                    st.success(f"**{predicted_class_name_st}** (Confiança: {confidence_score:.2%})")

            with col_pred_xai:
                st.subheader("Explicações da IA (XAI)")
                tab_gradcam, tab_shap, tab_lime = st.tabs(["Grad-CAM", "SHAP", "LIME"])

                with tab_gradcam:
                    if last_conv_name_st and last_conv_name_st != "conv2d_fallback_placeholder":
                        with st.spinner("Gerando Grad-CAM..."):
                            try:
                                heatmap_gradcam_st = make_gradcam_heatmap_st(processed_img_for_model_st, model_st_obj,
                                                                             last_conv_name_st,
                                                                             base_model_name_st_gradcam)
                                superimposed_gradcam_st = display_gradcam_st(original_img_uint8_st, heatmap_gradcam_st)
                                fig_grad_st, ax_grad_st = plt.subplots()
                                ax_grad_st.imshow(superimposed_gradcam_st)
                                ax_grad_st.set_title(f"Grad-CAM: Foco para '{predicted_class_name_st}'")
                                ax_grad_st.axis('off')
                                st.pyplot(fig_grad_st)
                                plt.close(fig_grad_st)
                                st.caption(
                                    "Grad-CAM destaca as regiões da imagem que o modelo considerou mais importantes.")
                            except Exception as e_grad:
                                st.error(f"Erro ao gerar Grad-CAM: {e_grad}")
                                st.exception(e_grad)
                    else:
                        st.warning(
                            "Grad-CAM: Não foi possível determinar a camada convolucional para heatmap ou ocorreu erro na seleção.")

                with tab_shap:
                    if shap_explainer_st_obj:
                        with st.spinner("Gerando SHAP (pode levar um momento)..."):
                            try:
                                shap_values_st = shap_explainer_st_obj.shap_values(processed_img_for_model_st)
                                shap_values_to_plot = shap_values_st[0] if isinstance(shap_values_st,
                                                                                      list) else shap_values_st
                                img_for_shap_display_st = original_img_uint8_st.astype(np.float32) / 255.0
                                if shap_values_to_plot.ndim == 3:
                                    shap_values_to_plot = np.expand_dims(shap_values_to_plot, axis=0)
                                fig_shap_st, ax_shap_st = plt.subplots(figsize=(7, 7))
                                shap.image_plot(
                                    shap_values_to_plot,
                                    np.expand_dims(img_for_shap_display_st, axis=0),
                                    show=False,
                                    ax=ax_shap_st
                                )
                                ax_shap_st.set_title(f"SHAP: Influência para '{predicted_class_name_st}'")
                                st.pyplot(fig_shap_st)
                                plt.close(fig_shap_st)
                                st.caption("SHAP: Vermelho = aumenta prob. da classe predita, Azul = diminui.")
                            except Exception as e_shap:
                                st.error(f"Erro ao gerar SHAP: {e_shap}")
                                st.exception(e_shap)
                    else:
                        st.info("SHAP Explainer não disponível.")

                with tab_lime:
                    if lime_explainer_st_obj:
                        with st.spinner("Gerando LIME (pode levar um momento)..."):
                            try:
                                def lime_predict_fn_st(images_lime_uint8):
                                    images_lime_float = images_lime_uint8.astype('float32')
                                    images_lime_processed = mobilenet_v2_preprocess_input(images_lime_float.copy())
                                    preds_lime = model_st_obj.predict(images_lime_processed, verbose=0)
                                    return np.hstack((1 - preds_lime, preds_lime))

                                explanation_lime_st = lime_explainer_st_obj.explain_instance(
                                    original_img_uint8_st,
                                    classifier_fn=lime_predict_fn_st,
                                    top_labels=1,
                                    hide_color=0,
                                    num_features=10,
                                    num_samples=300,
                                    random_seed=42
                                )
                                temp_lime_st, mask_lime_st = explanation_lime_st.get_image_and_mask(
                                    explanation_lime_st.top_labels[0],
                                    positive_only=False,
                                    num_features=10,
                                    hide_rest=False
                                )
                                fig_lime_st, ax_lime_st = plt.subplots(figsize=(7, 7))
                                ax_lime_st.imshow(mark_boundaries(temp_lime_st / 2 + 0.5, mask_lime_st))
                                ax_lime_st.set_title(f"LIME: Superpixels para '{predicted_class_name_st}'")
                                ax_lime_st.axis('off')
                                st.pyplot(fig_lime_st)
                                plt.close(fig_lime_st)
                                st.caption("LIME: Verde = contribuição positiva, Vermelho = negativa.")
                            except Exception as e_lime:
                                st.error(f"Erro ao gerar LIME: {e_lime}")
                                st.exception(e_lime)
                    else:
                        st.info("LIME Explainer não disponível.")
        elif model_st_obj is not None:
            st.info("⬅️ Carregue uma imagem de MRI na barra lateral para começar a análise.")

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    model_trained_or_loaded = None
    # Verifica se o Streamlit está rodando para não executar o treino múltiplas vezes desnecessariamente
    # se o modelo já existe, a menos que FORCE_TRAIN_MODEL seja True.
    is_streamlit_running = STREAMLIT_AVAILABLE and hasattr(st,
                                                           '_is_running_with_streamlit') and st._is_running_with_streamlit

    should_train = not os.path.exists(MODEL_SAVE_PATH) or FORCE_TRAIN_MODEL

    if should_train:
        print(
            f"INFO: Modelo '{MODEL_SAVE_PATH}' não encontrado ou treinamento forçado (FORCE_TRAIN_MODEL={FORCE_TRAIN_MODEL}).")
        if is_streamlit_running:
            # No Streamlit, só treine se o modelo realmente não existir ou forçado,
            # e evite treinar repetidamente em re-runs se o modelo já foi treinado nesta sessão.
            # Usar st.session_state para controlar se o treino já foi tentado/concluído nesta sessão.
            if 'model_training_attempted_this_session' not in st.session_state or FORCE_TRAIN_MODEL:
                with st.spinner(
                        "Treinamento do modelo em progresso... Isto pode levar vários minutos. A página será recarregada ao concluir."):
                    model_trained_or_loaded = train_and_save_model()
                st.session_state.model_training_attempted_this_session = True  # Marcar que o treino foi tentado
                if model_trained_or_loaded is not None or os.path.exists(MODEL_SAVE_PATH):
                    st.rerun()  # Recarregar para usar o novo modelo
                else:
                    st.error("Falha no treinamento do modelo. Verifique os logs do console.")
            # Se já tentou treinar e falhou, ou se o modelo foi treinado e a página recarregou,
            # não tenta treinar de novo a menos que FORCE_TRAIN_MODEL seja True.
        else:  # Rodando via console Python
            model_trained_or_loaded = train_and_save_model()

        if model_trained_or_loaded is None and not os.path.exists(MODEL_SAVE_PATH):
            fatal_msg = "ERRO FATAL: Falha no treinamento e nenhum modelo pré-existente encontrado. Saindo."
            print(fatal_msg)
            if is_streamlit_running:
                st.error(fatal_msg + " Verifique os logs do console.")
            # exit(1) # Comentado para permitir que o Streamlit continue mostrando o erro na UI
    else:
        print(f"INFO: Usando modelo existente: {MODEL_SAVE_PATH}")

    if STREAMLIT_AVAILABLE:
        # Se 'model_training_attempted_this_session' não está definido, e o modelo não existe,
        # a interface pode não ter sido chamada ainda se o treino falhou e o script saiu.
        # A lógica acima tenta garantir que run_streamlit_app_interface seja chamada.
        run_streamlit_app_interface()
    else:
        print("\nINFO: Bibliotecas Streamlit não encontradas. A interface web não será iniciada.")
        print("Se o treinamento ocorreu, o modelo (se bem-sucedido) está salvo.")
        print(f"Para iniciar a interface web (se as bibliotecas forem instaladas), execute no terminal:")
        print(f"streamlit run {os.path.basename(__file__)}")