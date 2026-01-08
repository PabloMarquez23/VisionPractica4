from ultralytics import YOLO
import torch

def entrenar():
    # Forzar el uso de GPU si está disponible
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("ADVERTENCIA: No se detectó GPU. Reinstala PyTorch con CUDA.")
    
    model = YOLO('yolo11n-seg.pt')

    # Iniciar el entrenamiento (Fine-tuning)
    model.train(
        data='data.yaml',      
        epochs=15,             
        imgsz=320,             
        device=device,         
        project='Practica_4',  
        name='segmentacion_personas'
    )

if __name__ == '__main__':
    entrenar()