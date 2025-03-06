import os
import json
import shutil
import random

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convierte un bounding box en formato COCO [x, y, w, h] a formato YOLO:
    [x_center, y_center, w, h] (con coordenadas normalizadas).
    """
    x, y, w, h = bbox
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    # Normalización
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

def create_dirs(base_output_dir):
    """
    Crea la siguiente estructura de directorios:
      base_output_dir/
          images/
              train/
              val/
              test/
          labels/
              train/
              val/
              test/
    """
    splits = ['train', 'val', 'test']
    images_dir = os.path.join(base_output_dir, 'images')
    labels_dir = os.path.join(base_output_dir, 'labels')
    for split in splits:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    return images_dir, labels_dir

def generate_dataset_yaml(output_dir, class_names):
    """
    Genera el archivo dataset.yaml con la siguiente estructura:
    
    train: <ruta a imágenes de entrenamiento>
    val: <ruta a imágenes de validación>
    test: <ruta a imágenes de prueba>
    nc: <número de clases>
    names: [lista de nombres]
    """
    yaml_content = f"""train: {os.path.join(output_dir, 'images', 'train')}
val: {os.path.join(output_dir, 'images', 'val')}
test: {os.path.join(output_dir, 'images', 'test')}

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Archivo dataset.yaml generado en: {yaml_path}")

def main():
    # Directorio base donde se encuentran las imágenes (las carpetas batch_*)
    base_dir = 'data'  # Ajusta si es necesario

    # Directorio de salida (se utilizará para crear la estructura final: images/ & labels/)
    output_dir = 'data'  # Se crearán subcarpetas dentro de este directorio

    # Cargar los archivos JSON
    with open('annotations.json', 'r') as f:
        annotations_data = json.load(f)
    with open('annotations_unofficial.json', 'r') as f:
        unofficial_data = json.load(f)

    # Extraer categorías y nombres de clases del archivo unofficial
    categories = unofficial_data.get('categories', [])
    # Ordenar las categorías por id (asumimos que id inician en 0 y son consecutivos)
    categories_sorted = sorted(categories, key=lambda c: c['id'])
    class_names = [cat['name'] for cat in categories_sorted]
    # Crear un mapeo: category_id original -> índice secuencial (0,1,2,...)
    category_mapping = {cat['id']: i for i, cat in enumerate(categories_sorted)}

    # Extraer la información de las imágenes (formato COCO)
    images = annotations_data.get('images', [])
    image_dict = {img['id']: img for img in images}

    # Extraer las anotaciones y agruparlas por image_id
    annotations_list = annotations_data.get('annotations', [])
    image_annotations = {}
    for ann in annotations_list:
        img_id = ann['image_id']
        image_annotations.setdefault(img_id, []).append(ann)

    # Listar todos los id de imágenes y barajarlos aleatoriamente
    image_ids = list(image_dict.keys())
    random.shuffle(image_ids)
    total = len(image_ids)
    train_count = int(0.7 * total)
    val_count = int(0.2 * total)
    test_count = total - train_count - val_count

    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count:train_count+val_count]
    test_ids = image_ids[train_count+val_count:]

    print(f"Total imágenes: {total}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Crear la estructura de directorios de salida
    images_out_dir, labels_out_dir = create_dirs(output_dir)

    def process_split(split_ids, split_name):
        for img_id in split_ids:
            img_info = image_dict[img_id]
            # Ruta original de la imagen (se asume que file_name incluye la carpeta, por ejemplo, batch_1/000006.jpg)
            orig_img_path = os.path.join(base_dir, img_info['file_name'])
            # Ruta destino: se copia la imagen a data/images/<split>/
            dest_img_path = os.path.join(images_out_dir, split_name, os.path.basename(img_info['file_name']))
            if os.path.exists(orig_img_path):
                shutil.copy(orig_img_path, dest_img_path)
            else:
                print(f"¡No se encontró la imagen: {orig_img_path}!")
                continue

            # Generar el archivo de etiquetas correspondiente (misma base del nombre con extensión .txt)
            label_lines = []
            anns = image_annotations.get(img_id, [])
            for ann in anns:
                # bbox en COCO: [x, y, w, h]
                bbox = ann['bbox']
                x_center, y_center, w, h = convert_bbox_to_yolo(bbox, img_info['width'], img_info['height'])
                # Remapear el category_id original a un índice secuencial
                orig_cat = ann['category_id']
                yolo_cat = category_mapping.get(orig_cat, orig_cat)
                label_lines.append(f"{yolo_cat} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            # Guardar el archivo de etiquetas (una línea por objeto)
            label_filename = os.path.splitext(os.path.basename(img_info['file_name']))[0] + '.txt'
            label_file_path = os.path.join(labels_out_dir, split_name, label_filename)
            with open(label_file_path, 'w') as lf:
                lf.write("\n".join(label_lines))
    
    print("Procesando conjunto de entrenamiento...")
    process_split(train_ids, 'train')
    print("Procesando conjunto de validación...")
    process_split(val_ids, 'val')
    print("Procesando conjunto de prueba...")
    process_split(test_ids, 'test')

    # Generar el archivo dataset.yaml para YOLOv5
    generate_dataset_yaml(output_dir, class_names)
    print("Preparación del dataset completada.")

if __name__ == '__main__':
    main()
