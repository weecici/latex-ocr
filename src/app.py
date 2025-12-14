import sys
import os
import subprocess
import shutil
import torch
import io
import time
import pickle as pkl
from typing import Literal
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QRubberBand,
    QComboBox,
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QAction, QIcon, QPen
from PIL import Image
from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet
from crnn import Im2LatexModel, Config, Vocab, inference


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL: Literal["hgnet", "crnn"] = "hgnet"


class SnippingWidget(QWidget):
    def __init__(self, parent=None, callback=None, pixmap=None):
        super().__init__(parent)
        self.callback = callback
        self.pixmap = pixmap
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start_point = None
        self.end_point = None

        # Make it cover all screens
        total_rect = QApplication.primaryScreen().geometry()
        for screen in QApplication.screens():
            total_rect = total_rect.united(screen.geometry())
        self.setGeometry(total_rect)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.pixmap:
            # Draw the screenshot
            painter.drawPixmap(0, 0, self.pixmap)
            # Draw a semi-transparent black overlay to dim the screen
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        else:
            # Fallback if no screenshot: just black overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()

            # Draw the selected region clearly (undimmed)
            if self.pixmap:
                painter.drawPixmap(rect, self.pixmap, rect)

            # Draw border
            pen = QPen(Qt.GlobalColor.red, 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = self.start_point
        self.update()

    def mouseMoveEvent(self, event):
        self.end_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.close()

        rect = QRect(self.start_point, self.end_point).normalized()
        if rect.width() > 5 and rect.height() > 5:
            if self.pixmap:
                # Crop from the captured pixmap
                cropped = self.pixmap.copy(rect)
                if self.callback:
                    self.callback(cropped)
            else:
                # Fallback: try to grab again (unlikely to work if initial failed)
                screen = QApplication.primaryScreen()
                pixmap = screen.grabWindow(
                    0, rect.x(), rect.y(), rect.width(), rect.height()
                )
                if self.callback:
                    self.callback(pixmap)


class LatexOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LaTeX OCR")
        self.setGeometry(100, 100, 800, 600)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.current_image = None  # PIL Image
        self.model_to_use: Literal["hgnet", "crnn"] = DEFAULT_MODEL

        self.init_ui()
        self.load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controls
        btn_layout = QHBoxLayout()

        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.btn_upload)

        self.btn_snip = QPushButton("Screenshot")
        self.btn_snip.clicked.connect(self.start_snip)
        btn_layout.addWidget(self.btn_snip)

        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItem("HGNet (FormulaNet)", userData="hgnet")
        self.model_selector.addItem("CRNN", userData="crnn")
        self.model_selector.setCurrentIndex(0)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        btn_layout.addWidget(QLabel("Model:"))
        btn_layout.addWidget(self.model_selector)

        self.btn_run = QPushButton("Run Inference")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_run.setStyleSheet("background-color: lightblue; font-weight: bold;")
        btn_layout.addWidget(self.btn_run)

        layout.addLayout(btn_layout)

        # Image Display
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px dashed gray; background-color: #f0f0f0;"
        )
        self.image_label.setMinimumHeight(300)
        layout.addWidget(self.image_label, stretch=1)

        # Result Display
        layout.addWidget(QLabel("Prediction:"))
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(100)
        self.result_text.setFontPointSize(12)
        layout.addWidget(self.result_text)

    def on_model_changed(self, index: int):
        # Update selected model and reload
        data = self.model_selector.itemData(index)
        if data in ("hgnet", "crnn"):
            self.model_to_use = data
            self.load_model()

    def load_model(self):
        self.result_text.setText("Loading model...")
        QApplication.processEvents()
        try:
            if self.model_to_use == "hgnet":
                path = "alephpi/FormulaNet"
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model = VisionEncoderDecoderModel.from_pretrained(path)
                self.model.to(self.device)
                self.model.eval()
            elif self.model_to_use == "crnn":

                self.args = Config()
                with open(self.args.vocab_path, "rb") as f:
                    self.vocab: Vocab = pkl.load(f)

                vocab_size = len(self.vocab)

                self.model = Im2LatexModel(
                    vocab_size,
                    self.args.word_emb_dim,
                    self.args.rnn_h_dim,
                    self.args.rnn_o_dim,
                    self.args.enc_out_dim,
                    self.args.att_dim,
                    self.args.dropout,
                )

                model_path = "models/crnn.pt"
                ckpt = torch.load(model_path, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
            else:
                raise ValueError(f"Unknown model type: {self.model_to_use}")

            self.result_text.setText("Model loaded successfully.")
        except Exception as e:
            self.result_text.setText(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            try:
                img = Image.open(file_path)
                self.set_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open image: {e}")

    def start_snip(self):
        # Check if we are on Hyprland or if grim/slurp are available
        is_wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland"
        has_grim_slurp = shutil.which("grim") and shutil.which("slurp")

        if is_wayland and has_grim_slurp:
            self.showMinimized()
            # Give time for minimize animation
            QTimer.singleShot(300, self._run_grim_slurp)
            return

        self.showMinimized()
        # Delay slightly to allow minimize animation
        QApplication.processEvents()
        QTimer.singleShot(300, self.take_screenshot_and_show_snipper)

    def _run_grim_slurp(self):
        try:
            # Run slurp to get geometry
            # This blocks until user selects a region
            slurp_process = subprocess.run(["slurp"], capture_output=True, text=True)

            if slurp_process.returncode != 0:
                # User cancelled
                self.showNormal()
                return

            geometry = slurp_process.stdout.strip()
            if not geometry:
                self.showNormal()
                return

            # Run grim with the geometry
            grim_process = subprocess.run(
                ["grim", "-g", geometry, "-"], capture_output=True
            )

            if grim_process.returncode == 0:
                image_data = grim_process.stdout
                img = Image.open(io.BytesIO(image_data))
                self.set_image(img)
            else:
                QMessageBox.warning(
                    self, "Error", "Failed to capture screenshot with grim."
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Screenshot failed: {e}")
        finally:
            self.showNormal()
            self.activateWindow()

    def take_screenshot_and_show_snipper(self):
        screen = QApplication.primaryScreen()
        # Grab the root window (0)
        pixmap = screen.grabWindow(0)

        self.snipper = SnippingWidget(callback=self.on_snip_captured, pixmap=pixmap)
        self.snipper.show()

    def on_snip_captured(self, pixmap):
        self.showNormal()
        self.activateWindow()

        # Convert QPixmap to PIL Image
        qimage = pixmap.toImage()
        buffer = qimage.bits().asstring(qimage.sizeInBytes())

        if qimage.format() == QImage.Format.Format_RGB32:
            img = Image.frombuffer(
                "RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1
            )
            img = img.convert("RGB")
        elif qimage.format() == QImage.Format.Format_ARGB32:
            img = Image.frombuffer(
                "RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1
            )
            img = img.convert("RGB")
        else:
            # Fallback via bytes
            byte_array = io.BytesIO()
            pixmap.save(byte_array, "PNG")
            img = Image.open(byte_array)

        self.set_image(img)

    def set_image(self, img):
        self.current_image = img.convert("RGB")

        # Display
        # Convert PIL to QPixmap for display
        im_data = self.current_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(
            im_data,
            self.current_image.size[0],
            self.current_image.size[1],
            QImage.Format.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(qim)

        # Scale if too large
        if (
            pixmap.width() > self.image_label.width()
            or pixmap.height() > self.image_label.height()
        ):
            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

    def run_inference(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return

        if self.model is None:
            QMessageBox.critical(self, "Error", "Model not loaded.")
            return

        self.result_text.setText("Running inference...")
        QApplication.processEvents()

        try:
            if self.model_to_use == "hgnet":
                image_processor = EvalMERImageProcessor(
                    image_size={"width": 384, "height": 384}
                )
                processed_image = image_processor(self.current_image).unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values=processed_image.to(self.device)
                    )

                pred_str = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
                self.result_text.setText(pred_str)
            elif self.model_to_use == "crnn":
                self.result_text.setText(
                    inference(self.model, self.vocab, self.current_image, self.device)
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_to_use}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Inference failed: {e}")
            self.result_text.setText(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LatexOCRApp()
    window.show()
    sys.exit(app.exec())
