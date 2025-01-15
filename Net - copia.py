import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QMessageBox, QInputDialog, QListWidget, QListWidgetItem, QDialog, QHBoxLayout, QTextEdit
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QEvent
import matplotlib.pyplot as plt
from skimage import exposure, measure, morphology, filters, restoration
import tifffile as tiff
import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
from skimage.io import imread
from skimage.io import imsave

class ImageSegmentationApp(QMainWindow):

    # Splash Screen
    def show_splash_screen(self):
        splash = tk.Tk()
        splash.title("alb-AI")
        splash.overrideredirect(True)  # Oculta la barra de título
        splash.geometry("500x500")  # Tamaño de la ventana de splash

        # Centrar la ventana en la pantalla
        screen_width = splash.winfo_screenwidth()
        screen_height = splash.winfo_screenheight()
        x = (screen_width // 2) - (500 // 2)
        y = (screen_height // 2) - (500 // 2)
        splash.geometry(f"500x500+{x}+{y}")

        # Añadir el logo
        logo_path = "alb-AI_logo.png"  # Cambia esto al nombre del archivo de tu logo
        img = Image.open(logo_path).resize((500, 500), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        label = tk.Label(splash, image=logo)
        label.image = logo  # Mantén una referencia para evitar el garbage collection
        label.pack()

        # Mostrar el splash screen por 3 segundos
        splash.after(3000, splash.destroy)
        splash.mainloop()

    def __init__(self):
        super().__init__()
        self.image_paths = []  # Lista para almacenar las rutas de las imágenes cargadas
        self.deleted_images = []  # Lista para almacenar las imágenes eliminadas
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Segmentation App")
        self.setGeometry(100, 100, 800, 600)

        # Establecer el icono de la ventana
        icon_path = 'icono.png'  # Cambia 'icono.png' por el nombre de tu archivo de icono
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Main layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)

        # Image display
        self.imageLabel = QLabel("No image loaded")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.imageLabel)

        # Image list
        self.imageListWidget = QListWidget()
        self.imageListWidget.itemClicked.connect(self.show_preview_option_for_selected_image)
        self.layout.addWidget(self.imageListWidget)

        # Buttons layout
        button_layout = QHBoxLayout()

        self.loadButton = QPushButton("Load Image")
        self.loadButton.clicked.connect(self.load_image)
        button_layout.addWidget(self.loadButton)

        self.trashButton = QPushButton("Open Trash")
        self.trashButton.clicked.connect(self.open_trash)
        button_layout.addWidget(self.trashButton)

        help_button = QPushButton("?")
        help_button.clicked.connect(self.show_main_help)
        button_layout.addWidget(help_button)

        self.layout.addLayout(button_layout)

        self.image = None
        self.segmented_image = None
        self.segmented_props = None

        # Install event filter to detect clicks outside the list
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            if source is self and not self.imageListWidget.underMouse():
                self.imageListWidget.clearSelection()
        return super().eventFilter(source, event)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.tif *.tiff)")
        if file_path:
            self.image_paths.append(file_path)  # Almacenar la ruta de la imagen cargada
            self.imageListWidget.addItem(QListWidgetItem(os.path.basename(file_path)))  # Agregar nombre de archivo a la lista
            self.update_image_label()
            self.show_preview_option(file_path)

    def update_image_label(self):
        if self.image_paths:
            self.imageLabel.setText("Select an image")
        else:
            self.imageLabel.setText("No image loaded")

    def show_preview_option(self, file_path):
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview Image in ImageJ")
        dialog.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout(dialog)

        preview_button = QPushButton("Preview in ImageJ")
        preview_button.clicked.connect(lambda: self.preview_in_imagej(file_path))
        layout.addWidget(preview_button)

        delete_button = QPushButton("Delete Image")
        delete_button.clicked.connect(lambda: self.delete_image(file_path, dialog))
        layout.addWidget(delete_button)

        load_button = QPushButton("Substack")
        load_button.clicked.connect(lambda: self.process_image(file_path, dialog))
        layout.addWidget(load_button)

        help_button = QPushButton("?")
        help_button.clicked.connect(self.show_preview_help)
        layout.addWidget(help_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_preview_option_for_selected_image(self, item):
        index = self.imageListWidget.row(item)
        if index != -1:
            file_path = self.image_paths[index]
            self.show_preview_option(file_path)

    def preview_in_imagej(self, file_path):
        # Asegúrate de que la ruta a ImageJ/Fiji sea correcta
        imagej_path = "C:\\Users\\albme\\Downloads\\fiji-win64\\Fiji.app\\ImageJ-win64.exe"  
        subprocess.run([imagej_path, file_path])

    def load_selected_image(self, item):
        index = self.imageListWidget.row(item)
        if index != -1:
            file_path = self.image_paths[index]
            self.process_image(file_path)

    def process_image(self, file_path, parent_dialog=None):
        if file_path.lower().endswith('.tif') or file_path.lower().endswith('.tiff'):
            layer_range, ok = QInputDialog.getText(self, "Layer Selection", "Enter start and end layers (comma separated):")
            if ok:
                start_layer, end_layer = map(int, layer_range.split(','))
                image = tiff.imread(file_path)
                substack = image[start_layer:end_layer]
                self.image = np.max(substack, axis=0)  # Proyección máxima
                self.show_image_dialog(self.image)
        else:
            self.image = imread(file_path)
            if self.image.ndim == 3:  # Convert RGB to grayscale if needed
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            self.show_image_dialog(self.image)
        if parent_dialog:
            parent_dialog.close()

    def show_image_dialog(self, image):
        dialog = QDialog(self)
        dialog.setWindowTitle("Image Preview & Options")
        dialog.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(dialog)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        height, width = image.shape
        bytes_per_line = width
        q_image = QPixmap.fromImage(QImage(np.uint8(image), width, height, bytes_per_line, QImage.Format_Grayscale8))
        image_label.setPixmap(q_image.scaled(image_label.size(), Qt.KeepAspectRatio))
        layout.addWidget(image_label)

        button_layout = QHBoxLayout()
        
        preprocess_button = QPushButton("Preprocess Image")
        preprocess_button.clicked.connect(lambda: self.show_preprocessing_dialog(image, dialog))
        button_layout.addWidget(preprocess_button)
        
        segment_button = QPushButton("Segment Image")
        segment_button.clicked.connect(lambda: self.segment_image_and_display(image, dialog))
        button_layout.addWidget(segment_button)
        
        visualize_button = QPushButton("Visualize & Inspect")
        visualize_button.clicked.connect(self.visualize_and_inspect)
        button_layout.addWidget(visualize_button)

        save_button = QPushButton("Save Segmented Image")
        save_button.clicked.connect(self.save_image)
        button_layout.addWidget(save_button)

        help_button = QPushButton("?")
        help_button.clicked.connect(self.show_image_help)
        button_layout.addWidget(help_button)
        
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def show_preprocessing_dialog(self, image, parent_dialog):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Preprocessing Option")
        dialog.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout(dialog)

        denoise_button = QPushButton("Denoise Image")
        denoise_button.clicked.connect(lambda: self.apply_preprocessing(image, dialog, "denoise"))
        layout.addWidget(denoise_button)
        
        enhance_button = QPushButton("Enhanced_Image")
        enhance_button.clicked.connect(lambda: self.apply_preprocessing(image, dialog, "enhance"))
        layout.addWidget(enhance_button)
        
        decon_button = QPushButton("Deconvolution")
        decon_button.clicked.connect(lambda: self.apply_preprocessing(image, dialog, "deconvolution"))
        layout.addWidget(decon_button)
        
        help_button = QPushButton("?")
        help_button.clicked.connect(self.show_preprocessing_help)
        layout.addWidget(help_button)
        
        layout.addStretch()
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_preprocessing(self, image, dialog, option):
        if option == "denoise":
            preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
        elif option == "enhance":
            preprocessed_image = exposure.equalize_adapthist(image, clip_limit=0.03)
        elif option == "deconvolution":
            psf = np.ones((5, 5)) / 25  # Aquí puedes ajustar el filtro
            preprocessed_image = restoration.richardson_lucy(image, psf, num_iter=30)
        
        self.display_image(preprocessed_image)
        dialog.close()

    def segment_image_and_display(self, image, dialog):
        thresh = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.segmented_image = morphology.remove_small_objects(thresh.astype(bool), min_size=50).astype(np.uint8)
        self.segmented_props = measure.regionprops(measure.label(self.segmented_image))
        self.display_image(self.segmented_image)
        dialog.close()

    def preprocess_image(self, image):
        # Reducción de ruido
        denoised = cv2.GaussianBlur(image, (5, 5), 0)

        # Aumento de contraste
        enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.03)

        return denoised, enhanced

    def segment_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        _, enhanced = self.preprocess_image(self.image)
        thresh = cv2.threshold((enhanced * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.segmented_image = morphology.remove_small_objects(thresh.astype(bool), min_size=50).astype(np.uint8)
        self.segmented_props = measure.regionprops(measure.label(self.segmented_image))
        self.display_image(self.segmented_image)

    def visualize_and_inspect(self):
        if self.segmented_image is None:
            QMessageBox.warning(self, "Warning", "Please segment an image first.")
            return

        def on_click(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.ydata), int(event.xdata)
                label_id = self.segmented_image[x, y]
                if label_id > 0:
                    props = self.segmented_props[label_id - 1]
                    info = f"Label: {label_id}\nArea: {props.area}\nPerimeter: {props.perimeter:.2f}"
                    QMessageBox.information(self, "Region Properties", info)

        fig, ax = plt.subplots()
        ax.imshow(self.segmented_image, cmap="gray")
        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()

    def save_image(self):
        if self.segmented_image is None:
            QMessageBox.warning(self, "Warning", "Please segment an image first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Segmented Image", "", "Images (*.png *.jpg *.tiff)")
        if file_path:
            imsave(file_path, (self.segmented_image > 0).astype(np.uint8) * 255)
            QMessageBox.information(self, "Success", "Segmented image saved successfully.")

    def display_image(self, img):
        height, width = img.shape
        bytes_per_line = width
        q_image = QPixmap.fromImage(QImage(np.uint8(img), width, height, bytes_per_line, QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(q_image.scaled(self.imageLabel.size(), Qt.KeepAspectRatio))

    def delete_image(self, file_path, parent_dialog):
        index = self.image_paths.index(file_path)
        self.deleted_images.append(self.image_paths.pop(index))
        self.imageListWidget.takeItem(index)
        self.update_image_label()
        parent_dialog.close()

    def open_trash(self):
        dialog = TrashDialog(self)
        dialog.exec_()

    def restore_image(self, trash_list_widget, parent_dialog):
        selected_items = trash_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select an image to restore.")
            return
        for item in selected_items:
            file_name = item.text()
            file_path = next((path for path in self.deleted_images if os.path.basename(path) == file_name), None)
            if file_path:
                self.deleted_images.remove(file_path)
                self.image_paths.append(file_path)
                self.imageListWidget.addItem(QListWidgetItem(file_name))
                trash_list_widget.takeItem(trash_list_widget.row(item))
        if not self.deleted_images:
            parent_dialog.close()
        self.update_image_label()

    def show_main_help(self):
        self.show_help_dialog("Main Menu Help", "This is the main menu. You can load images, open the trash, and access help.")

    def show_preview_help(self):
        self.show_help_dialog("Preview Menu Help", "In this menu, you can preview images in ImageJ, delete images, or create a substack.")

    def show_image_help(self):
        self.show_help_dialog("Image Options Help", "In this menu, you can preprocess, segment, visualize, and save images.")

    def show_preprocessing_help(self):
        self.show_help_dialog("Preprocessing Help", "In this menu, you can choose to denoise, enhance contrast, or apply deconvolution to the image.")

    def show_help_dialog(self, title, message):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(dialog)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setText(message)
        layout.addWidget(help_text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

class TrashDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trash")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout(self)

        self.trash_list_widget = QListWidget()
        for file_path in parent.deleted_images:
            self.trash_list_widget.addItem(QListWidgetItem(os.path.basename(file_path)))
        self.layout.addWidget(self.trash_list_widget)

        button_layout = QHBoxLayout()

        preview_button = QPushButton("Preview in ImageJ")
        preview_button.clicked.connect(lambda: parent.preview_in_imagej(self.trash_list_widget.currentItem().text()))
        button_layout.addWidget(preview_button)

        restore_button = QPushButton("Restore Image")
        restore_button.clicked.connect(lambda: parent.restore_image(self.trash_list_widget, self))
        button_layout.addWidget(restore_button)

        help_button = QPushButton("?")
        help_button.clicked.connect(self.show_trash_help)
        button_layout.addWidget(help_button)

        self.layout.addLayout(button_layout)

        # Install event filter to detect clicks outside the list
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            if source is self and not self.trash_list_widget.underMouse():
                self.trash_list_widget.clearSelection()
        return super().eventFilter(source, event)

    def show_trash_help(self):
        self.show_help_dialog("Trash Help", "In this menu, you can preview and restore deleted images.")

    def show_help_dialog(self, title, message):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(dialog)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setText(message)
        layout.addWidget(help_text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == "__main__":
    # Mostrar Splash Screen antes de cargar la aplicación principal
    app = QApplication(sys.argv)
    icon_path = 'icono.png'  # Cambia 'icono.png' por el nombre de tu archivo de icono
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))  # Configurar el icono a nivel de aplicación
    mainWindow = ImageSegmentationApp()
    mainWindow.show_splash_screen()
    mainWindow.show()
    sys.exit(app.exec_())