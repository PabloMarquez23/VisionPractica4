import cv2
import torch
import time
import psutil
from ultralytics import YOLO

# 1. CARGA DEL MODELO (Red a)
model_yolo = YOLO('runs/segment/train/weights/best.pt') 

# 2. CONFIGURACIÓN DE DISPOSITIVO (Prueba con 0 y luego con 'cpu')
DEVICE = 0 # 0 para GPU MX230, 'cpu' para procesador

# 3. INICIALIZACIÓN DE WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam.")
    exit()

print(f"Iniciando comparativa en: {DEVICE}. Presiona 'q' para salir.")

while True:
    start_time = time.time() # Inicio para cálculo de FPS
    
    ret, frame = cap.read()
    if not ret: break

    # --- RED (a): YOLOv11 Segmentación ---
    # show=False es CRITICO para que no se abran mil ventanas
    results = model_yolo.predict(source=frame, device=DEVICE, imgsz=320, conf=0.5, show=False, verbose=False)
    
    # Dibujamos los resultados en el frame original
    annotated_frame = results[0].plot()

    # --- RED (b): Super-Resolución (Simulación de carga) ---
    # Escalamos la imagen para aplicar carga de procesamiento (Cubic Interpolation)
    alto, ancho = annotated_frame.shape[:2]
    sr_frame = cv2.resize(annotated_frame, (ancho*2, alto*2), interpolation=cv2.INTER_CUBIC)

    # --- CÁLCULO DE MÉTRICAS (Requisitos Guía Parte 1B) ---
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    ram_uso = psutil.virtual_memory().percent

    # 4. TEXTO EN PANTALLA PARA EL VÍDEO
    cv2.putText(sr_frame, f"FPS: {fps:.2f} | RAM: {ram_uso}%", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(sr_frame, f"Device: {DEVICE} | MAC: 8c:c8:4b:f5:2b:31", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 5. MOSTRAR UNA SOLA VENTANA
    cv2.imshow("Practica 1B - Comparativa Final", sr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()