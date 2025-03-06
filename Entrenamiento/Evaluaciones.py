
from ultralytics import YOLO


def evaluar_modelo(modelo, data_yaml):
   
    model = YOLO(modelo)  

    
    resultados = model.val(data=data_yaml)  
   
    metrics = resultados.results_dict

   
    print("Claves disponibles en los resultados:", metrics.keys())

  
    if 'mAP_0.5' in metrics:
        print(f"mAP@0.5: {metrics['mAP_0.5']:.4f}")
    if 'mAP_0.5:0.95' in metrics:
        print(f"mAP@0.5:0.95: {metrics['mAP_0.5:0.95']:.4f}")
    if 'precision' in metrics:
        print(f"Precisión: {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"Recuerdo (Recall): {metrics['recall']:.4f}")
    if 'f1' in metrics:
        print(f"F1: {metrics['f1']:.4f}")


modelo_path = "runs/detect/train/weights/best.pt"  


dataset_yaml = "data/dataset.yaml" 


evaluar_modelo(modelo_path, dataset_yaml)
