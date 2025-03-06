import flet as ft
import threading
import time
import io
import tempfile
import os
from datetime import datetime
import base64
from PIL import Image

from supabase import create_client
from ultralytics import YOLO

# Configuración de Supabase
SUPABASE_URL = "https://xshchsisefefyazmgewl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzaGNoc2lzZWZlZnlhem1nZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NTAwNjYsImV4cCI6MjA1NDAyNjA2Nn0.OGxuHQ_ApZpC27APq2GdpXuMtACyTqOkwr-DXzC4lT4"
SOURCE_BUCKET = "objetos"
DESTINATION_BUCKET = "deteccion"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = YOLO("best2.pt")


def procesar_imagen(img_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(img_bytes)
        tmp_path = tmp_file.name
    results = model(tmp_path)
    result_img = results[0].plot()
    image = Image.fromarray(result_img[:, :, ::-1])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def get_images(bucket):
    files = supabase.storage.from_(bucket).list()
    return [f["name"] for f in files if f["name"].endswith(("jpg", "jpeg", "png"))]


def cargar_imagenes(bucket):
    """
    Descarga las imágenes del bucket y devuelve una lista de cadenas base64 con el prefijo de Data URI.
    """
    imagenes_names = get_images(bucket)
    lista_imagenes = []
    for nombre in imagenes_names:
        try:
            file_data = supabase.storage.from_(bucket).download(nombre)
            base64_image = base64.b64encode(file_data).decode("utf-8")
            # Incluimos el prefijo Data URI para JPEG
            lista_imagenes.append(f"data:image/jpeg;base64,{base64_image}")
        except Exception as ex:
            print(f"Error cargando {nombre}: {ex}")
    return lista_imagenes


def create_carousel(bucket):
    images_list = cargar_imagenes(bucket)
    if not images_list:
        return ft.Text("❌ No hay imágenes disponibles ❌", color="red")
    
    
    current_index = 0
    initial_image = images_list[current_index].replace("data:image/jpeg;base64,", "")
    image_control = ft.Image(
        src_base64=initial_image,
        width=500,
        height=300,
        animate_opacity=ft.Animation(600, "ease-in-out"),
        opacity=1.0,
        fit=ft.ImageFit.CONTAIN
    )

    def change_image(offset):
        nonlocal current_index

        def animate_transition():
            nonlocal current_index
            image_control.opacity = 0.0  
            image_control.update()
            time.sleep(0.6) 
            current_index = (current_index + offset) % len(images_list)
            new_image = images_list[current_index].replace("data:image/jpeg;base64,", "")
            image_control.src_base64 = new_image
            image_control.opacity = 1.0  
            image_control.update()

        threading.Thread(target=animate_transition, daemon=True).start()

    # Botones para navegar en el carrusel
    prev_button = ft.IconButton(icon=ft.icons.ARROW_BACK_IOS_NEW, on_click=lambda e: change_image(-1))
    next_button = ft.IconButton(icon=ft.icons.ARROW_FORWARD_IOS, on_click=lambda e: change_image(1))

    carousel = ft.Row(
        [prev_button, image_control, next_button],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20,
    )
    return carousel


def app(page: ft.Page):
    page.title = "Nuance Vision"
    page.window_width = 300
    page.window_height = 300
    page.bgcolor = "#121212"
    
    # Agregar icono de la ventana
    page.window_icon = "logo.ico" 
    
    # Agregar icono de la ventana
    page.window_icon = "logo.ico"  # Actualiza esta ruta con tu icono

    # Agregar cabecera con logos
    logo = ft.Image(src="logo.png", width=120, height=120, fit=ft.ImageFit.CONTAIN)  
    header = ft.Row(
        controls=[
            logo,
        
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )

    # Consola para el tab "Ejecutar"
    console = ft.Column([], expand=True)

    # Contenedor para el carrusel de imágenes (se actualizará dinámicamente)
    carousel_container = ft.Column(
        controls=[create_carousel(DESTINATION_BUCKET)],
        expand=True,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    # Gif de carga
    loading_gif = ft.Image(
        src="katarina.gif", 
        width=100, 
        height=100, 
        opacity=0.0  # Inicialmente oculto
    )
    loading_gif_container = ft.Column([console, loading_gif], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def ejecutar(e):
        console.controls.clear()
        console.controls.append(ft.Text("➤➤➤ Iniciando procesamiento ➤➤➤", weight=ft.FontWeight.BOLD, color="white70"))
        page.update()

        # Muestra la barra de carga
        loading_gif.opacity = 1.0
        page.update()

        imagenes = get_images(SOURCE_BUCKET)
        if not imagenes:
            console.controls.append(ft.Text("❌ No se encontraron imágenes ❌", weight=ft.FontWeight.BOLD, color="red900"))
        else:
            for nombre in imagenes:
                try:
                    file_data = supabase.storage.from_(SOURCE_BUCKET).download(nombre)
                    resultado = procesar_imagen(file_data)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    new_name = f"deteccion_{os.path.splitext(nombre)[0]}_{timestamp}.jpg"
                    supabase.storage.from_(DESTINATION_BUCKET).upload(new_name, resultado)
                    supabase.storage.from_(SOURCE_BUCKET).remove([nombre])
                    console.controls.append(ft.Text(f"⏩ Procesado: {new_name}", color="green"))
                except Exception as ex:
                    console.controls.append(ft.Text(f"Error con {nombre}: {str(ex)}", weight=ft.FontWeight.BOLD, color="red"))
        
        # Oculta el gif al finalizar
        loading_gif.opacity = 0.0
        console.controls.append(ft.Text("✅ Fin de la ejecución ✅", weight=ft.FontWeight.BOLD, color="green"))
        
        # Actualiza el carrusel en todos los casos
        carousel_container.controls.clear()
        carousel_container.controls.append(create_carousel(DESTINATION_BUCKET))
        page.update()

    # Tab "Ejecutar" con consola y botón de acción
    ejecutar_tab = ft.Tab(
        text="Ejecutar",
        content=ft.Column(
            [loading_gif_container, ft.ElevatedButton("Ejecutar", on_click=ejecutar)],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

    # Tab "Imágenes" con el contenedor del carrusel actualizado
    imagenes_tab = ft.Tab(
        text="Imágenes",
        content=carousel_container
    )

    tab_view = ft.Tabs(
        selected_index=0,
        expand=True,
        tabs=[ejecutar_tab, imagenes_tab]
    )

    # Se añade la cabecera y los tabs a la página
    page.add(header, tab_view)


ft.app(target=app)
