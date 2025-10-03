#!/usr/bin/env python3
"""
Versão FUNCIONAL e ESTÁVEL do TraduLibras
Sistema de reconhecimento INCLUSAO BC sem travamentos
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import threading
import time
import subprocess
import json
from auth import user_manager, User

app = Flask(__name__)
app.secret_key = 'tradulibras_secret_key_2024'

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor, faça login para acessar esta página.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return user_manager.get_user(user_id)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Carregar modelo INCLUSAO BC
modelo_inclusao_bc = 'modelos/modelo_inclusao_bc_20251003_144506.pkl'
scaler_inclusao_bc = 'modelos/scaler_inclusao_bc_20251003_144506.pkl'
info_inclusao_bc = 'modelos/modelo_info_inclusao_bc_20251003_144506.pkl'

if os.path.exists(modelo_inclusao_bc) and os.path.exists(scaler_inclusao_bc) and os.path.exists(info_inclusao_bc):
    with open(modelo_inclusao_bc, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_inclusao_bc, 'rb') as f:
        scaler = pickle.load(f)
    with open(info_inclusao_bc, 'rb') as f:
        model_info = pickle.load(f)
    print(f"✅ Modelo INCLUSAO BC carregado com sucesso!")
    print(f"📊 Classes: {model_info['classes']}")
else:
    print("❌ ERRO: Modelo INCLUSAO BC não encontrado!")
    model = None
    scaler = None
    model_info = {'classes': [], 'accuracy': 0}

# Variáveis globais
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 2.5  # Tempo entre detecções (reduzido)
hand_detected_time = None
min_hand_time = 1.5  # 1.5 segundos após detectar mão (reduzido)

def process_landmarks(hand_landmarks):
    """Processar landmarks da mão"""
    if not hand_landmarks:
        return None
    
    # Ponto de referência (pulso)
    wrist = hand_landmarks.landmark[0]
    
    # Features básicas
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y
        ])
    
    # Features extras
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Distâncias entre dedos e pulso
    features.extend([
        abs(thumb_tip.x - wrist.x) + abs(thumb_tip.y - wrist.y),
        abs(index_tip.x - wrist.x) + abs(index_tip.y - wrist.y),
        abs(middle_tip.x - wrist.x) + abs(middle_tip.y - wrist.y),
        abs(ring_tip.x - wrist.x) + abs(ring_tip.y - wrist.y),
        abs(pinky_tip.x - wrist.x) + abs(pinky_tip.y - wrist.y)
    ])
    
    # Distâncias entre dedos
    features.extend([
        abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y),
        abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y),
        abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y),
        abs(ring_tip.x - pinky_tip.x) + abs(ring_tip.y - pinky_tip.y)
    ])
    
    return features  # Total: 51 features

def generate_frames():
    """Gerar frames da câmera ESTÁVEL"""
    global current_letter, formed_text, corrected_text, last_prediction_time, hand_detected_time
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Câmera inicializada e funcionando!")
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip horizontal
        frame = cv2.flip(frame, 1)
        
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = hands.process(rgb_frame)
        
        # Variável para landmarks
        points = None
        current_time = datetime.now()
        
        # Interface visual
        # SEM TEXTOS NA JANELA DA CÂMERA - INTERFACE COMPLETAMENTE LIMPA
        
        if results.multi_hand_landmarks:
            # Se mão detectada pela primeira vez
            if hand_detected_time is None:
                hand_detected_time = current_time
                print("👋 Mão detectada! Aguardando estabilização...")
            
            # Desenhar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Processar landmarks
                points = process_landmarks(hand_landmarks)
            
            # Mostrar status
            time_since_detection = (current_time - hand_detected_time).total_seconds()
            
            if time_since_detection < min_hand_time:
                # Aguardando estabilização - limpar letra atual
                current_letter = ""  # Limpar letra durante estabilização
                # SEM TEXTOS NA JANELA DA CÂMERA
            else:
                # Pronto para detectar - SEM TEXTOS NA JANELA DA CÂMERA
                
                # Processar letra se passar do tempo mínimo E cooldown
                time_since_last = (current_time - last_prediction_time).total_seconds()
                
                if time_since_last >= prediction_cooldown and points and len(points) == 51:
                    try:
                        if model and scaler:
                            # Normalizar features
                            points_normalized = scaler.transform([points])
                            prediction = model.predict(points_normalized)
                            predicted_letter = prediction[0]
                            
                            # Processar letra detectada
                            if predicted_letter == 'ESPACO':
                                current_letter = '[ESPAÇO]'
                                formed_text += ' '
                                corrected_text = formed_text
                            else:
                                current_letter = predicted_letter
                                formed_text += predicted_letter
                                corrected_text = formed_text
                            
                            # Lógica de finalização removida - detecção contínua
                            
                            # Atualizar tempo da última predição
                            last_prediction_time = current_time
                            hand_detected_time = None  # Reset para próxima detecção
                            
                            print(f"✅ Letra detectada: {predicted_letter}")
                            
                    except Exception as e:
                        print(f"❌ Erro na predição: {e}")
        else:
            # Sem mão detectada - limpar letra atual
            hand_detected_time = None
            current_letter = ""  # Limpar letra quando não há mão
            # SEM TEXTOS NA JANELA DA CÂMERA
        
        # SEM TEXTOS NA JANELA DA CÂMERA - INTERFACE COMPLETAMENTE LIMPA
        # Não mostrar nenhum texto na câmera
        
        # Converter frame para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpeg = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = user_manager.authenticate(username, password)
        
        if user:
            login_user(user)
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Usuário ou senha incorretos!')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Você foi desconectado.')
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    user_stats = user_manager.get_stats()
    return render_template('admin_dashboard.html', user_stats=user_stats)

@app.route('/camera')
@login_required
def camera_tradulibras():
    return render_template('camera_tradulibras.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/letra_atual')
def get_current_letter():
    return jsonify({"letra": current_letter, "texto": formed_text})

@app.route('/limpar_texto', methods=['GET', 'POST'])
@login_required
def limpar_texto():
    global formed_text, corrected_text, current_letter
    formed_text = ""
    corrected_text = ""
    current_letter = ""
    return jsonify({"status": "success", "message": "Texto limpo com sucesso"})

@app.route('/limpar_ultima_letra', methods=['POST'])
@login_required
def limpar_ultima_letra():
    global formed_text, current_letter
    if formed_text:
        formed_text = formed_text[:-1]  # Remove a última letra
        current_letter = ""
        return jsonify({"status": "success", "message": "Última letra removida", "texto": formed_text})
    else:
        return jsonify({"status": "error", "message": "Não há texto para limpar"})

@app.route('/corrigir_letra', methods=['POST'])
@login_required
def corrigir_letra():
    global formed_text, current_letter
    data = request.get_json()
    nova_letra = data.get('letra', '').strip()
    
    if nova_letra and formed_text:
        # Remove a última letra e adiciona a nova
        formed_text = formed_text[:-1] + nova_letra
        current_letter = nova_letra
        return jsonify({"status": "success", "texto": formed_text, "letra": current_letter})
    else:
        return jsonify({"status": "error", "message": "Letra inválida ou texto vazio"})

@app.route('/falar_texto', methods=['GET', 'POST'])
@login_required
def falar_texto():
    global formed_text
    
    if formed_text.strip():
        try:
            # Usar gTTS para gerar o áudio
            tts = gTTS(text=formed_text, lang='pt-br', slow=False)
            
            # Criar arquivo temporário único
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            temp_file = os.path.join(temp_dir, f'speech_{timestamp}.mp3')
            
            # Salvar áudio
            tts.save(temp_file)
            
            # Retornar o arquivo de áudio como resposta via send_file
            return send_file(temp_file, mimetype='audio/mpeg', as_attachment=False)
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "Nenhum texto para falar"})

@app.route('/status')
@login_required
def status():
    return jsonify({
        "modelo_carregado": model is not None,
        "classes": model_info.get('classes', []),
        "acuracia": model_info.get('accuracy', 0),
        "texto_atual": formed_text,
        "letra_atual": current_letter
    })

if __name__ == '__main__':
    print("🚀 TRADULIBRAS FUNCIONAL - INCLUSAO BC")
    print("=" * 50)
    print(f"📊 Classes suportadas: {model_info.get('classes', [])}")
    print(f"🎯 Acesso: http://localhost:5000")
    print("=" * 50)
    print("✅ Sistema ESTÁVEL e funcionando!")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
