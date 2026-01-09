import cv2
import torch
import time
import uuid
import psutil  # Librería para medir la RAM del sistema
import os
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
DEVICE = 'cpu'  # Alternar: 'cuda' o 'cpu'
MODO_PRUEBA = 'ambas' # Alternar: 'yolo', 'sr', o 'ambas'

# Obtener MAC Address (Identidad Única)
mac_addr = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0,8*6,8)][::-1])

# Carga Red (a): YOLOv11
print(f"Iniciando Benchmark en {DEVICE.upper()}...")
model = YOLO('yolo11n-seg.pt').to(DEVICE)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    start_time = time.time()

    # --- RED (a): YOLOv11 SEGMENTACIÓN ---
    if MODO_PRUEBA in ['yolo', 'ambas']:
        results = model.predict(frame, device=DEVICE, verbose=False)
        frame = results[0].plot()

    # --- RED (b): SUPER RESOLUCIÓN (Carga Computacional) ---
    if MODO_PRUEBA in ['sr', 'ambas']:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

    end_time = time.time()
    latencia = (end_time - start_time) * 1000
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

    # --- MÉTRICAS DE MEMORIA ---
    # 1. RAM del Sistema (Usada por el proceso actual)
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2  # Convertir a MB

    # 2. VRAM de la GPU (Solo si CUDA está activo)
    vram_info = "0.0 MB"
    if DEVICE == 'cuda' and torch.cuda.is_available():
        vram_info = f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"

    # --- PANEL DE INFORMACIÓN ---
    cv2.rectangle(frame, (10, 10), (480, 180), (0,0,0), -1) # Fondo para datos
    
    cv2.putText(frame, f"MAC: {mac_addr}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.2f} | Latencia: {latencia:.1f}ms", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"RAM Proceso: {ram_usage:.1f} MB", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    cv2.putText(frame, f"VRAM Reservada: {vram_info}", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Device: {DEVICE.upper()} | Modo: {MODO_PRUEBA}", (20, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Parte 1B: Benchmark Hardware Completo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()