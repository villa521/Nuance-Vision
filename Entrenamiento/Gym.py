from ultralytics import YOLO

def main():
    # Ruta al archivo YAML que define el dataset (imágenes y etiquetas)
    data_yaml = 'data/dataset.yaml'  # Ajusta la ruta a tu archivo dataset.yaml
 
    # Ruta a los pesos preentrenados del modelo YOLOv5 (si no lo tienes, se descargará automáticamente)
    weights = 'best2.pt'  # Ajusta a la ruta de tus pesos preentrenados

    # Crea una instancia del modelo. Se cargarán los pesos y la arquitectura correspondiente.
    model = YOLO(weights)

  # Inicia el entrenamiento:
    model.train(
        data=data_yaml,  # Ruta al archivo dataset.yaml
        epochs=100,         # Número de épocas de entrenamiento (ajústalo según tus necesidades)
        imgsz=640,        # Tamaño de las imágenes de entrada (416 es un buen punto de partida)
        batch=16,          # Tamaño del batch (si tu CPU es limitada, reduce este valor)
        device='cpu',      # Usamos 'cpu' si no tienes GPU. Si tienes GPU, usa 'cuda'
         lr0=0.01             
    )

if __name__ == '__main__':
    main()

