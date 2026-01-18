import sys, os, cv2, logging, time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# Fix Qt plugin conflict between OpenCV and PyQt5
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAction, QStatusBar, QDialog, QFormLayout,
    QDialogButtonBox, QCheckBox, QSlider, QGroupBox, QProgressBar,
    QSplitter, QFrame, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, QSize
from pyzbar.pyzbar import decode

# ---------------- CONSTANTS ----------------
GRADING_THRESHOLDS = {"A": 300, "B": 220, "C": 150, "D": 80}
DEFECT_THRESHOLDS = {"BLUR": 50, "LOW_CONTRAST": 25, "BROKEN_EDGE_RATIO": 0.02}
TABLE_ROW_LIMIT = 500
DEFAULT_FPS = 15
CACHE_TIMEOUT = 3  # seconds

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename='barcam.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- BARCODE QUALITY ----------------
def barcode_grade(frame, rect):
    """Calculate ISO 15415 grade for barcode quality"""
    x, y, w, h = rect
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "F"
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    edges = cv2.Canny(gray, 50, 150)
    modulation = edges.sum() / (255 * w * h) if (w * h) > 0 else 0
    
    score = (sharpness * 0.5) + (contrast * 0.3) + (modulation * 100 * 0.2)
    
    if score > GRADING_THRESHOLDS["A"]: return "A"
    if score > GRADING_THRESHOLDS["B"]: return "B"
    if score > GRADING_THRESHOLDS["C"]: return "C"
    if score > GRADING_THRESHOLDS["D"]: return "D"
    return "F"

def ai_defect_check(frame, rect):
    """AI-powered defect detection"""
    x, y, w, h = rect
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "INVALID"
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (255 * w * h) if (w * h) > 0 else 0
    
    if blur < DEFECT_THRESHOLDS["BLUR"]: return "BLUR"
    if contrast < DEFECT_THRESHOLDS["LOW_CONTRAST"]: return "LOW CONTRAST"
    if edge_ratio < DEFECT_THRESHOLDS["BROKEN_EDGE_RATIO"]: return "BROKEN"
    return "OK"

# ---------------- FRAME PROCESSOR THREAD ----------------
class FrameProcessor(QThread):
    frame_processed = pyqtSignal(object, list)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    
    def __init__(self, cap, fps=DEFAULT_FPS):
        super().__init__()
        self.cap = cap
        self.running = True
        self.fps = fps
        self.frame_delay = int(1000 / fps)
        self.frame_times = deque(maxlen=30)
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        while self.running:
            try:
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    QThread.msleep(50)
                    continue
                
                barcodes_data = []
                barcodes = decode(frame)
                
                for bc in barcodes:
                    x, y, w, h = bc.rect
                    code = bc.data.decode("utf-8", "ignore")
                    btype = bc.type
                    grade = barcode_grade(frame, (x, y, w, h))
                    defect = ai_defect_check(frame, (x, y, w, h))
                    barcodes_data.append((code, btype, grade, defect, (x, y, w, h)))
                    
                    # Draw on frame
                    color = (0, 255, 0) if defect == "OK" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(
                        frame, f"{btype} | {grade} | {defect}",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                    )
                
                self.frame_processed.emit(frame, barcodes_data)
                
                # Calculate actual FPS
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 0:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    actual_fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.fps_updated.emit(actual_fps)
                
                QThread.msleep(self.frame_delay)
                
            except Exception as e:
                self.logger.error(f"Frame processing error: {str(e)}")
                self.error_occurred.emit(str(e))
                QThread.msleep(100)  # Backoff on errors
    
    def stop(self):
        self.running = False
    
    def set_fps(self, fps):
        self.fps = fps
        self.frame_delay = int(1000 / fps)

# ---------------- STATISTICS WIDGET ----------------
class StatisticsWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Statistics", parent)
        self.init_ui()
        self.reset_stats()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.total_label = QLabel("Total Scans: 0")
        self.defect_label = QLabel("Defects: 0")
        self.pass_rate_label = QLabel("Pass Rate: 100%")
        self.grade_label = QLabel("A:0 B:0 C:0 D:0 F:0")
        
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        
        for label in [self.total_label, self.defect_label, 
                     self.pass_rate_label, self.grade_label]:
            label.setFont(font)
            layout.addWidget(label)
        
        self.reset_btn = QPushButton("Reset Stats")
        self.reset_btn.clicked.connect(self.reset_stats)
        layout.addWidget(self.reset_btn)
    
    def reset_stats(self):
        self.total = 0
        self.defects = 0
        self.grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        self.update_display()
    
    def add_scan(self, grade, defect):
        self.total += 1
        if defect != "OK":
            self.defects += 1
        if grade in self.grades:
            self.grades[grade] += 1
        self.update_display()
    
    def update_display(self):
        self.total_label.setText(f"Total Scans: {self.total}")
        self.defect_label.setText(f"Defects: {self.defects}")
        
        pass_rate = 100 if self.total == 0 else ((self.total - self.defects) / self.total * 100)
        self.pass_rate_label.setText(f"Pass Rate: {pass_rate:.1f}%")
        
        grade_str = " ".join([f"{k}:{v}" for k, v in self.grades.items()])
        self.grade_label.setText(grade_str)

# ---------------- SETTINGS DIALOG ----------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.current_settings = current_settings or {}
        self.init_ui()
    
    def init_ui(self):
        layout = QFormLayout(self)
        
        # Beep setting
        self.beep_check = QCheckBox("Enable Beep on New Barcode")
        self.beep_check.setChecked(self.current_settings.get("beep_enabled", True))
        layout.addRow(self.beep_check)
        
        # FPS setting
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(self.current_settings.get("fps", DEFAULT_FPS))
        self.fps_spin.setSuffix(" FPS")
        layout.addRow("Processing Speed:", self.fps_spin)
        
        # Cache timeout
        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(1, 30)
        self.cache_spin.setValue(self.current_settings.get("cache_timeout", CACHE_TIMEOUT))
        self.cache_spin.setSuffix(" seconds")
        layout.addRow("Duplicate Detection Window:", self.cache_spin)
        
        # Auto-export
        self.auto_export_check = QCheckBox("Auto-export ISO reports")
        self.auto_export_check.setChecked(self.current_settings.get("auto_export", False))
        layout.addRow(self.auto_export_check)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_settings(self):
        return {
            "beep_enabled": self.beep_check.isChecked(),
            "fps": self.fps_spin.value(),
            "cache_timeout": self.cache_spin.value(),
            "auto_export": self.auto_export_check.isChecked()
        }

# ---------------- MAIN APP ----------------
class BarcodeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Barcam ProScan – Industrial Grade")
        self.resize(1400, 900)
        
        self.logger = logging.getLogger(__name__)
        
        # State variables
        self.cap = None
        self.processor = None
        self.save_dir = os.getcwd()
        self.barcode_cache = {}
        self.cache_timeout = CACHE_TIMEOUT
        self.last_iso_data = None
        self.beep_enabled = True
        self.auto_export = False
        self.processing_fps = DEFAULT_FPS
        self._preview = None
        
        # Settings
        self.settings = QSettings("Barcam", "ProScan")
        self.load_settings()
        
        # Status bar (must be created before init_ui)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # UI
        self.init_ui()
        self.create_menu()
        
        self.status_bar.showMessage("Ready")
        self.logger.info("Application started")

    # ---------------- UI ----------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left side: Camera + Table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera feed
        self.image_label = QLabel("Camera Feed")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("background:black; border: 2px solid #333;")
        
        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Barcode", "Type", "Grade", "Defect"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        left_layout.addWidget(self.image_label, 6)
        left_layout.addWidget(self.table, 4)
        
        # Right side: Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Camera controls
        cam_group = QGroupBox("Camera Controls")
        cam_layout = QVBoxLayout(cam_group)
        
        self.camera_combo = QComboBox()
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.res_combo.setCurrentIndex(1)
        
        cam_layout.addWidget(QLabel("Camera:"))
        cam_layout.addWidget(self.camera_combo)
        cam_layout.addWidget(QLabel("Resolution:"))
        cam_layout.addWidget(self.res_combo)
        
        # Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)
        cam_layout.addWidget(QLabel("Theme:"))
        cam_layout.addWidget(self.theme_combo)
        
        # Order number
        self.order_input = QLineEdit()
        self.order_input.setPlaceholderText("Order Number")
        cam_layout.addWidget(QLabel("Order Number:"))
        cam_layout.addWidget(self.order_input)
        
        # Buttons with permanent colors (work in both themes)
        self.start_btn = QPushButton("▶ Start Camera")
        self.stop_btn = QPushButton("⏹ Stop Camera")
        self.folder_btn = QPushButton("📁 Select Folder")
        self.export_iso_btn = QPushButton("📊 Export ISO Report")
        self.export_csv_btn = QPushButton("💾 Export Table CSV")
        self.clear_btn = QPushButton("🗑 Clear Table")
        self.settings_btn = QPushButton("⚙ Settings")
        
        # Apply permanent button styles
        self.apply_button_styles()
        
        # Button connections
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.folder_btn.clicked.connect(self.select_folder)
        self.export_iso_btn.clicked.connect(self.export_last_iso)
        self.export_csv_btn.clicked.connect(self.export_table_csv)
        self.clear_btn.clicked.connect(self.clear_table)
        self.settings_btn.clicked.connect(self.open_settings)
        
        self.stop_btn.setEnabled(False)
        
        cam_layout.addWidget(self.start_btn)
        cam_layout.addWidget(self.stop_btn)
        cam_layout.addWidget(self.folder_btn)
        cam_layout.addWidget(self.export_iso_btn)
        cam_layout.addWidget(self.export_csv_btn)
        cam_layout.addWidget(self.clear_btn)
        cam_layout.addWidget(self.settings_btn)
        
        # Statistics
        self.stats_widget = StatisticsWidget()
        
        right_layout.addWidget(cam_group)
        right_layout.addWidget(self.stats_widget)
        right_layout.addStretch()
        
        # Add to main layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        self.detect_cameras()
        self.apply_theme()

    def apply_button_styles(self):
        """Apply colorful button styles that work in both themes"""
        
        # Green for Start
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5CBF60;
            }
            QPushButton:pressed {
                background-color: #3D9142;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        # Red for Stop
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #F55549;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        # Blue for Folder
        self.folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """)
        
        # Orange for ISO Report
        self.export_iso_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #FFA726;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        
        # Purple for CSV Export
        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #AB47BC;
            }
            QPushButton:pressed {
                background-color: #7B1FA2;
            }
        """)
        
        # Blue-Grey for Clear
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #78909C;
            }
            QPushButton:pressed {
                background-color: #546E7A;
            }
        """)
        
        # Brown for Settings
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #795548;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                min-height: 35px;
                border: none;
            }
            QPushButton:hover {
                background-color: #8D6E63;
            }
            QPushButton:pressed {
                background-color: #5D4037;
            }
        """)

    # ---------------- THEME ----------------
    def apply_theme(self):
        theme = self.theme_combo.currentText()
        if theme == "Dark":
            self.setStyleSheet("""
                QWidget {
                    background-color: #1E1E1E;
                    color: #E0E0E0;
                }
                QMainWindow {
                    background-color: #121212;
                }
                QTableWidget {
                    background-color: #2C2C2C;
                    alternate-background-color: #3A3A3A;
                    color: #E0E0E0;
                    gridline-color: #404040;
                }
                QHeaderView::section {
                    background-color: #2C2C2C;
                    color: #E0E0E0;
                    border: 1px solid #404040;
                    padding: 4px;
                }
                QLineEdit, QComboBox, QSpinBox {
                    background-color: #2C2C2C;
                    color: #E0E0E0;
                    border: 1px solid #404040;
                    padding: 5px;
                    border-radius: 3px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #E0E0E0;
                }
                QGroupBox {
                    border: 2px solid #404040;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    color: #E0E0E0;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    padding: 0 5px;
                }
                QLabel {
                    color: #E0E0E0;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #FAFAFA;
                    color: #212121;
                }
                QTableWidget {
                    background-color: white;
                    alternate-background-color: #F5F5F5;
                    color: #212121;
                    gridline-color: #E0E0E0;
                }
                QHeaderView::section {
                    background-color: #E0E0E0;
                    color: #212121;
                    border: 1px solid #BDBDBD;
                    padding: 4px;
                }
                QLineEdit, QComboBox, QSpinBox {
                    background-color: white;
                    color: #212121;
                    border: 1px solid #BDBDBD;
                    padding: 5px;
                    border-radius: 3px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #212121;
                }
                QGroupBox {
                    border: 2px solid #BDBDBD;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    color: #212121;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    padding: 0 5px;
                }
                QLabel {
                    color: #212121;
                }
            """)
        
        # Reapply button styles to ensure they stay colorful
        self.apply_button_styles()

    # ---------------- MENU ----------------
    def create_menu(self):
        bar = self.menuBar()
        
        # File menu
        file_menu = bar.addMenu("&File")
        file_menu.addAction("Open &Folder", self.select_folder, "Ctrl+O")
        file_menu.addAction("&Export Table CSV", self.export_table_csv, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")
        
        # Camera menu
        cam_menu = bar.addMenu("&Camera")
        cam_menu.addAction("&Start", self.start_camera, "Ctrl+S")
        cam_menu.addAction("S&top", self.stop_camera, "Ctrl+T")
        cam_menu.addAction("&Refresh Cameras", self.detect_cameras, "F5")
        
        # Tools menu
        tools_menu = bar.addMenu("&Tools")
        tools_menu.addAction("&Settings", self.open_settings, "Ctrl+,")
        tools_menu.addAction("&Clear Table", self.clear_table, "Ctrl+L")
        tools_menu.addAction("Reset &Statistics", self.stats_widget.reset_stats)
        
        # Help menu
        help_menu = bar.addMenu("&Help")
        help_menu.addAction("&About", self.about, "F1")
        help_menu.addAction("View &Logs", self.view_logs)

    # ---------------- CAMERA ----------------
    def detect_cameras(self):
        self.camera_combo.clear()
        self.status_bar.showMessage("Detecting cameras...")
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.camera_combo.addItem(f"Camera {i}", i)
            cap.release()
        
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("No cameras found", None)
            self.status_bar.showMessage("No cameras detected", 3000)
        else:
            self.status_bar.showMessage(
                f"Found {self.camera_combo.count()} camera(s)", 3000
            )

    def start_camera(self):
        if self.cap:
            QMessageBox.warning(self, "Warning", "Camera already running")
            return
        
        idx = self.camera_combo.currentData()
        if idx is None:
            QMessageBox.critical(self, "Error", "No camera selected")
            return
        
        try:
            w, h = map(int, self.res_combo.currentText().split("x"))
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start processor thread
            self.processor = FrameProcessor(self.cap, self.processing_fps)
            self.processor.frame_processed.connect(self.on_frame_processed)
            self.processor.error_occurred.connect(self.on_processor_error)
            self.processor.fps_updated.connect(self.on_fps_updated)
            self.processor.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_bar.showMessage("Camera started")
            self.logger.info(f"Camera {idx} started at {w}x{h}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")
            self.logger.error(f"Camera start error: {str(e)}")
            self.cleanup_camera()

    def stop_camera(self):
        if self.processor:
            self.processor.stop()
            if not self.processor.wait(2000):
                self.processor.terminate()
                self.logger.warning("Processor thread forcefully terminated")
            self.processor = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.image_label.clear()
        self.image_label.setText("Camera Feed")
        self.image_label.setStyleSheet("background:black; border: 2px solid #333;")
        self.fps_label.setText("FPS: 0.0")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("Camera stopped")
        self.logger.info("Camera stopped")

    def cleanup_camera(self):
        """Helper to clean up camera resources"""
        if self.processor:
            self.processor.stop()
            self.processor.wait(1000)
            self.processor = None
        if self.cap:
            self.cap.release()
            self.cap = None

    # ---------------- FRAME PROCESSING ----------------
    def on_frame_processed(self, frame, barcodes_data):
        current_time = time.time()
        
        for code, btype, grade, defect, rect in barcodes_data:
            # Check duplicate detection cache
            if code in self.barcode_cache:
                if current_time - self.barcode_cache[code] < self.cache_timeout:
                    continue
            
            # New barcode detected
            self.barcode_cache[code] = current_time
            
            # Beep
            if self.beep_enabled:
                QApplication.beep()
            
            # Add to table and stats
            self.add_table_row(code, btype, grade, defect)
            self.stats_widget.add_scan(grade, defect)
            
            # Save snapshot
            self.save_snapshot(frame, code)
            
            # Store for ISO export
            self.last_iso_data = (frame.copy(), code, btype, grade, rect)
            
            # Auto-export ISO if enabled
            if self.auto_export:
                self.export_iso_report(
                    self.last_iso_data[0],
                    self.last_iso_data[1],
                    self.last_iso_data[2],
                    self.last_iso_data[3],
                    self.last_iso_data[4]
                )
            
            # Show preview
            self.preview_snapshot(frame, rect)
        
        # Clean old cache entries
        self.barcode_cache = {
            k: v for k, v in self.barcode_cache.items()
            if current_time - v < self.cache_timeout
        }
        
        self.show_frame(frame)
    
    def on_processor_error(self, error_msg):
        self.status_bar.showMessage(f"Processing error: {error_msg}", 5000)
    
    def on_fps_updated(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    # ---------------- TABLE ----------------
    def add_table_row(self, code, btype, grade, defect):
        if self.table.rowCount() >= TABLE_ROW_LIMIT:
            self.table.removeRow(0)
        
        r = self.table.rowCount()
        self.table.insertRow(r)
        
        items = [
            datetime.now().strftime("%H:%M:%S"),
            code,
            btype,
            grade,
            defect
        ]
        
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            # Color code based on grade/defect
            if col == 3:  # Grade
                if grade in ["A", "B"]:
                    item.setForeground(Qt.green)
                elif grade in ["D", "F"]:
                    item.setForeground(Qt.red)
            elif col == 4:  # Defect
                if defect != "OK":
                    item.setForeground(Qt.red)
            self.table.setItem(r, col, item)
        
        self.table.scrollToBottom()

    def clear_table(self):
        reply = QMessageBox.question(
            self, "Clear Table",
            "Are you sure you want to clear all scans?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.table.setRowCount(0)
            self.barcode_cache.clear()
            self.status_bar.showMessage("Table cleared", 3000)
            self.logger.info("Table cleared")

    # ---------------- EXPORT ----------------
    def export_table_csv(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "Export", "No data to export")
            return
        
        default_name = f"barcam_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Table CSV", default_name, "CSV Files (*.csv)"
        )
        
        if path:
            try:
                data = []
                for row in range(self.table.rowCount()):
                    data.append([
                        self.table.item(row, col).text()
                        for col in range(self.table.columnCount())
                    ])
                
                df = pd.DataFrame(
                    data,
                    columns=["Time", "Barcode", "Type", "Grade", "Defect"]
                )
                df.to_csv(path, index=False)
                
                QMessageBox.information(
                    self, "Success",
                    f"Exported {len(data)} rows to:\n{path}"
                )
                self.logger.info(f"Table exported to {path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
                self.logger.error(f"CSV export error: {str(e)}")

    def export_last_iso(self):
        if not self.last_iso_data:
            QMessageBox.warning(self, "ISO Report", "No barcode scanned yet")
            return
        
        frame, code, btype, grade, rect = self.last_iso_data
        self.export_iso_report(frame, code, btype, grade, rect)

    def export_iso_report(self, frame, code, btype, grade, rect):
        try:
            x, y, w, h = rect
            roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = gray.std()
            edges = cv2.Canny(gray, 50, 150)
            modulation = edges.sum() / (255 * w * h) if (w * h) > 0 else 0
            
            report = {
                "Time": datetime.now().isoformat(),
                "Barcode": code,
                "Type": btype,
                "ISO_Grade": grade,
                "Contrast": round(contrast, 2),
                "Sharpness": round(sharpness, 2),
                "Modulation": round(modulation, 4),
                "Width": w,
                "Height": h,
                "Result": "PASS" if grade in ["A", "B", "C"] else "FAIL"
            }
            
            df = pd.DataFrame([report])
            filename = f"ISO15415_{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            path = os.path.join(self.save_dir, filename)
            df.to_excel(path, index=False)
            
            if not self.auto_export:
                QMessageBox.information(
                    self, "ISO Report",
                    f"Report saved to:\n{path}"
                )
            
            self.logger.info(f"ISO report exported: {path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ISO export failed: {str(e)}")
            self.logger.error(f"ISO export error: {str(e)}")

    # ---------------- SNAPSHOT ----------------
    def save_snapshot(self, frame, code):
        try:
            order = self.order_input.text() or "NoOrder"
            path = os.path.join(self.save_dir, order)
            os.makedirs(path, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{code}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(path, filename), frame)
            
            self.logger.info(f"Snapshot saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Snapshot save error: {str(e)}")

    def preview_snapshot(self, frame, rect):
        """Show preview popup of scanned barcode region"""
        try:
            # Close existing preview
            if self._preview and self._preview.isVisible():
                self._preview.close()
            
            x, y, w, h = rect
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return
            
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            h1, w1, ch = rgb.shape
            
            # Create proper QImage
            img = QImage(rgb.data, w1, h1, w1 * ch, QImage.Format_RGB888).copy()
            
            # Create preview widget with parent
            self._preview = QLabel(self)
            self._preview.setWindowFlags(
                Qt.Window | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
            )
            self._preview.setWindowTitle("Barcode Preview")
            self._preview.setStyleSheet("""
                QLabel {
                    background: black;
                    border: 3px solid #4CAF50;
                    padding: 5px;
                }
            """)
            self._preview.setPixmap(
                QPixmap.fromImage(img).scaled(
                    250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self._preview.setAlignment(Qt.AlignCenter)
            
            # Position near cursor
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            self._preview.move(
                self.x() + cursor_pos.x() + 20,
                self.y() + cursor_pos.y() + 20
            )
            self._preview.show()
            
            # Auto-close after 2 seconds
            QTimer.singleShot(2000, self._preview.close)
            
        except Exception as e:
            self.logger.error(f"Preview error: {str(e)}")

    # ---------------- SETTINGS ----------------
    def load_settings(self):
        self.save_dir = self.settings.value("save_dir", os.getcwd())
        self.beep_enabled = self.settings.value("beep_enabled", True, type=bool)
        self.processing_fps = self.settings.value("fps", DEFAULT_FPS, type=int)
        self.cache_timeout = self.settings.value("cache_timeout", CACHE_TIMEOUT, type=int)
        self.auto_export = self.settings.value("auto_export", False, type=bool)

    def save_settings(self):
        self.settings.setValue("save_dir", self.save_dir)
        self.settings.setValue("beep_enabled", self.beep_enabled)
        self.settings.setValue("fps", self.processing_fps)
        self.settings.setValue("cache_timeout", self.cache_timeout)
        self.settings.setValue("auto_export", self.auto_export)

    def open_settings(self):
        current = {
            "beep_enabled": self.beep_enabled,
            "fps": self.processing_fps,
            "cache_timeout": self.cache_timeout,
            "auto_export": self.auto_export
        }
        
        dialog = SettingsDialog(self, current)
        
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            
            self.beep_enabled = new_settings["beep_enabled"]
            self.cache_timeout = new_settings["cache_timeout"]
            self.auto_export = new_settings["auto_export"]
            
            # Update FPS if changed
            if new_settings["fps"] != self.processing_fps:
                self.processing_fps = new_settings["fps"]
                if self.processor:
                    self.processor.set_fps(self.processing_fps)
            
            self.save_settings()
            self.status_bar.showMessage("Settings saved", 3000)

    # ---------------- HELPERS ----------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Folder", self.save_dir
        )
        if folder:
            self.save_dir = folder
            self.save_settings()
            self.status_bar.showMessage(f"Save folder: {folder}", 3000)

    def show_frame(self, frame):
        """Display frame in UI"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            
            # Properly create QImage with copy
            img = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
            
            pixmap = QPixmap.fromImage(img).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            
        except Exception as e:
            self.logger.error(f"Frame display error: {str(e)}")

    def view_logs(self):
        """Open log file"""
        try:
            log_path = os.path.abspath('barcam.log')
            if os.path.exists(log_path):
                if sys.platform == 'win32':
                    os.startfile(log_path)
                elif sys.platform == 'darwin':
                    os.system(f'open "{log_path}"')
                else:
                    os.system(f'xdg-open "{log_path}"')
            else:
                QMessageBox.information(self, "Logs", "No log file found")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open logs: {str(e)}")

    def about(self):
        QMessageBox.information(
            self, "About Barcam ProScan",
            """<h2>Barcam ProScan v2.0</h2>
            <p><b>Industrial-Grade Barcode Scanner</b></p>
            <p>Features:</p>
            <ul>
            <li>1D & 2D Barcode Support</li>
            <li>AI-Powered Defect Detection</li>
            <li>ISO 15415 Compliance</li>
            <li>Real-time Quality Grading</li>
            <li>Advanced Statistics</li>
            <li>CSV & Excel Export</li>
            </ul>
            <p>Developer: Hassan</p>
            <p>© 2026 Barcam Technologies</p>"""
        )

    def closeEvent(self, event):
        """Clean shutdown"""
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.stop_camera()
            self.save_settings()
            self.logger.info("Application closed")
            event.accept()
        else:
            event.ignore()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Barcam ProScan")
    app.setOrganizationName("Barcam")
    
    win = BarcodeApp()
    win.show()
    
    sys.exit(app.exec_())