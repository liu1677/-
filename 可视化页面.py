#!/usr/bin/env python3
from __future__ import annotations
import struct
import sys
import time
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Union

import cv2
import numpy as np
import serial
import torch
from PIL import Image, ImageDraw, ImageFont
from serial.tools import list_ports
from ultralytics import YOLO

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
COLOR_CODE_TO_CHAR = {
    'BLU': 'B',   
    'YEL': 'Y',   
  
    'RED': 'R',
    'ORG': 'O',
    'GRN': 'G',
    'CYN': 'C',
    'PUR': 'P',
    'WHT': 'W',
    'GRY': 'G',   
    'BLK': 'K',
    'UNK': 'A',
}


import math
camera_matrix = np.array([
    [496.8, 0, 305.29],
    [0, 495.02, 274.31],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([-0.480003616602357, 0.295570656773631, 
                       -0.000532223886668543, -0.000103897937464121, 
                       -0.107320244853150], dtype=np.float32)

cube_size = 0.03  



def calculate_joints(arm_x_cm: float, arm_y_cm: float) -> tuple[float|None, float|None, float|None, float|None]:
    """
    输入机械臂坐标系下的 (X, Y) 物理坐标 (cm)，返回 j1-j4 角度 (度)。
    基于 C 代码逻辑，使用底座半径 P 和连杆长度 a1~a4。
    请根据实际机械臂测量修改下面的参数。
    """
  

    P = 0     
    a1 = 10.0    
    a2 = 10     
    a3 = 9.5    
    a4 = 14.0     
    X = arm_x_cm
    Y = arm_y_cm
    Z = 8   
 
    if X == 0:
        j1 = 90.0
    else:
        j1 = math.degrees(math.atan((Y + P) / X))
    if j1 < 0:
        j1 = 180 + j1   

  
    n = 0
    feasible_solutions = []  
    for i in range(0, 181):
        j_all_rad = math.radians(i)
        len_xy = math.hypot(Y + P, X)          
        L = len_xy - a4 * math.sin(j_all_rad)
        H = Z - a4 * math.cos(j_all_rad) - a1

        # 计算 cos_j3
        cos_j3 = (L*L + H*H - a2*a2 - a3*a3) / (2 * a2 * a3)
        if abs(cos_j3) > 1.0:
            continue
        sin_j3 = math.sqrt(1 - cos_j3*cos_j3)
        j3 = math.degrees(math.atan2(sin_j3, cos_j3))   

        # 计算 j2
        K2 = a3 * math.sin(math.radians(j3))
        K1 = a2 + a3 * math.cos(math.radians(j3))
        denom = K1*K1 + K2*K2
        if denom == 0:
            continue
        cos_j2 = (K2 * L + K1 * H) / denom
        if abs(cos_j2) > 1.0:
            continue
        sin_j2 = math.sqrt(1 - cos_j2*cos_j2)
        j2 = math.degrees(math.atan2(sin_j2, cos_j2))

        j4 = i - j2 - j3


        if (0 <= j2 <= 180) and (0 <= j3 <= 180) and (-90 <= j4 <= 90):
            n += 1
            feasible_solutions.append((j2, j3, j4))

    if n == 0:
        return None, None, None, None

    idx = n // 2 if n % 2 == 0 else (n + 1) // 2
    j2, j3, j4 = feasible_solutions[idx - 1]

    return j1, j2, j3, j4

def map_to_servos(j1, j2, j3, j4):
    """映射为最终发送给舵机的 6 个角度"""
    return (j1, 180-j2, j3, 90.0 + j4, 90.0, 68.0)

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
QIMAGE_RGB888 = QImage.Format_RGB888
KEEP_ASPECT = Qt.KeepAspectRatio
SMOOTH_TRANSFORM = Qt.SmoothTransformation
HEADER_STRETCH = QHeaderView.Stretch
HEADER_RESIZE = QHeaderView.ResizeToContents
SELECT_ROWS = QAbstractItemView.SelectRows
NO_EDIT = QAbstractItemView.NoEditTriggers

COLOR_RULES = [
    ("红色", "RED", (70, 90, 255), [(0, 10), (165, 179)]),
    ("橙色", "ORG", (35, 140, 255), [(11, 22)]),
    ("黄色", "YEL", (0, 228, 255), [(23, 35)]),
    ("绿色", "GRN", (60, 210, 80), [(36, 85)]),
    ("青色", "CYN", (255, 215, 0), [(86, 102)]),
    ("蓝色", "BLU", (255, 120, 60), [(103, 135)]),
    ("紫色", "PUR", (180, 80, 220), [(136, 164)]),
]

NEUTRAL_COLORS = {
    "白色": ("WHT", (240, 240, 240)),
    "灰色": ("GRY", (160, 160, 160)),
    "黑色": ("BLK", (48, 48, 48)),
    "未知": ("UNK", (120, 120, 120)),
}

FONT_PATTERNS = [
    "PingFang*.ttc",
    "Hiragino Sans GB*.ttc",
    "STHeiti*.ttc",
    "Microsoft YaHei*.ttc",
    "msyh*.ttc",
    "simhei.ttf",
    "SimHei.ttf",
    "Deng*.ttc",
    "NotoSansCJK*.ttc",
    "NotoSansCJK*.otf",
    "SourceHanSans*.otf",
    "SourceHanSans*.ttf",
    "WenQuanYi*.ttc",
    "wqy-*.ttc",
]


@dataclass
class DetectionInfo:
    object_id: str
    class_name: str
    color_name: str
    color_code: str
    confidence: float
    area_ratio: float
    center_x: int
    center_y: int
    norm_u: float
    norm_v: float
    bbox: tuple[int, int, int, int]
    bgr: tuple[int, int, int]
    send_status: str = "待发送"


def rel_path(path: Path | None) -> str:
    if path is None:
        return "--"
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def short_text(text: str, limit: int = 22) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def discover_models() -> list[Path]:
    preferred = [ROOT / "runs/seg/weights/best.pt", ROOT / "runs/seg/weights/last.pt"]
    found: list[Path] = []
    seen: set[Path] = set()

    for path in preferred + sorted((ROOT / "runs").rglob("*.pt")) + sorted(ROOT.glob("*.pt")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        found.append(resolved)
        seen.add(resolved)
    return found


def discover_demo_image() -> Path | None:
    for pattern in ("datasets/images/test/*", "datasets/images/train/*", "labelme/*"):
        for path in sorted(ROOT.glob(pattern)):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                return path
    return None


def read_local_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_local_image(path: Path, frame: np.ndarray) -> bool:
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, frame)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QIMAGE_RGB888).copy()
    return QPixmap.fromImage(image)


def auto_device() -> tuple[str | int, str]:
    if torch.cuda.is_available():
        return 0, "CUDA"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps", "MPS"
    return "cpu", "CPU"


def _font_search_roots() -> list[Path]:
    roots = [
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
        Path("C:/Windows/Fonts"),
    ]
    return [root for root in roots if root.exists()]


@lru_cache(maxsize=1)
def find_cjk_font_path() -> Path | None:
    direct_candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf"),
        Path("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in direct_candidates:
        if path.is_file():
            return path

    for root in _font_search_roots():
        for pattern in FONT_PATTERNS:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


@lru_cache(maxsize=16)
def get_cjk_font(size: int) -> ImageFont.ImageFont:
    font_path = find_cjk_font_path()
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font_size: int,
    fill: tuple[int, int, int],
) -> None:
    draw.text(position, text, font=get_cjk_font(font_size), fill=fill)


def draw_tag(
    draw: ImageDraw.ImageDraw,
    canvas_size: tuple[int, int],
    text: str,
    x: int,
    y: int,
    border_color: tuple[int, int, int],
    *,
    font_size: int = 20,
) -> None:
    font = get_cjk_font(font_size)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    pad_x = 10
    pad_y = 6
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    box_x1 = min(max(0, x), max(0, canvas_size[0] - box_w - 1))
    box_y1 = min(max(0, y), max(0, canvas_size[1] - box_h - 1))
    box_x2 = min(canvas_size[0] - 1, box_x1 + box_w)
    box_y2 = min(canvas_size[1] - 1, box_y1 + box_h)
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return
    draw.rounded_rectangle((box_x1, box_y1, box_x2, box_y2), radius=8, fill=(9, 20, 30), outline=border_color, width=1)
    text_x = box_x1 + pad_x
    text_y = box_y1 + pad_y - text_box[1]
    draw.text((text_x, text_y), text, font=font, fill=(244, 250, 255))


def classify_color(frame: np.ndarray, mask: np.ndarray | None, bbox: tuple[int, int, int, int]) -> tuple[str, str, tuple[int, int, int]]:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        code, bgr = NEUTRAL_COLORS["未知"]
        return "未知", code, bgr

    if mask is not None and mask.any():
        pixels = frame[mask]
    else:
        pixels = frame[y1:y2, x1:x2].reshape(-1, 3)

    if pixels.size == 0:
        code, bgr = NEUTRAL_COLORS["未知"]
        return "未知", code, bgr

    if len(pixels) > 12000:
        sample_index = np.linspace(0, len(pixels) - 1, 12000).astype(int)
        pixels = pixels[sample_index]

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_values = hsv[:, 0]
    s_values = hsv[:, 1]
    v_values = hsv[:, 2]

    mean_s = float(np.mean(s_values))
    mean_v = float(np.mean(v_values))

    if mean_v < 42:
        code, bgr = NEUTRAL_COLORS["黑色"]
        return "黑色", code, bgr
    if mean_s < 28:
        if mean_v > 205:
            code, bgr = NEUTRAL_COLORS["白色"]
            return "白色", code, bgr
        code, bgr = NEUTRAL_COLORS["灰色"]
        return "灰色", code, bgr

    valid = (s_values > 40) & (v_values > 45)
    best_name = "未知"
    best_code = "UNK"
    best_bgr = (120, 120, 120)
    best_score = 0

    for name, code, bgr, ranges in COLOR_RULES:
        score = 0
        for low, high in ranges:
            score += int(np.count_nonzero((h_values >= low) & (h_values <= high) & valid))
        if score > best_score:
            best_name = name
            best_code = code
            best_bgr = bgr
            best_score = score

    if best_score < max(50, int(len(hsv) * 0.08)):
        code, bgr = NEUTRAL_COLORS["未知"]
        return "未知", code, bgr
    return best_name, best_code, best_bgr


def render_overlay(frame: np.ndarray, detections: list[DetectionInfo], source_tag: str, device_label: str, latency_ms: float) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (352, 118), (6, 16, 26), -1)
    output = cv2.addWeighted(overlay, 0.82, output, 0.18, 0)
    cv2.rectangle(output, (12, 12), (352, 118), (95, 211, 255), 1)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), det.bgr, 2)
        cv2.circle(output, (det.center_x, det.center_y), 4, det.bgr, -1)

    image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    draw_text(draw, (24, 22), "智能分拣视觉", font_size=24, fill=(244, 250, 255))
    draw_text(draw, (24, 48), f"来源 {source_tag}", font_size=18, fill=(220, 239, 250))
    draw_text(draw, (24, 70), f"设备 {device_label}", font_size=18, fill=(220, 239, 250))
    draw_text(draw, (24, 92), f"目标 {len(detections)}   延迟 {latency_ms:.1f} ms", font_size=18, fill=(220, 239, 250))

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label_y = max(0, y1 - 34)
        center_y = min(image.height - 34, y2 + 8)
        border_rgb = (det.bgr[2], det.bgr[1], det.bgr[0])
        draw_tag(draw, image.size, f"{det.color_name} {det.color_code} {det.confidence:.2f}", x1, label_y, border_rgb, font_size=18)
        draw_tag(draw, image.size, f"中心 ({det.center_x}, {det.center_y})", x1, center_y, border_rgb, font_size=18)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def analyze_result(frame: np.ndarray, result: Any, source_tag: str, device_label: str, latency_ms: float) -> dict[str, Any]:
    detections: list[DetectionInfo] = []
    height, width = frame.shape[:2]
    image_area = max(1, height * width)

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    names = getattr(result, "names", {}) or {}
    mask_data = getattr(masks, "data", None) if masks is not None else None

    if boxes is not None:
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
            class_name = str(names.get(class_id, "block"))

            mask = None
            if mask_data is not None and index < len(mask_data):
                mask = mask_data[index].detach().cpu().numpy()
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
                area_ratio = float(np.count_nonzero(mask)) / image_area
            else:
                area_ratio = float(max(0, x2 - x1) * max(0, y2 - y1)) / image_area

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            color_name, color_code, bgr = classify_color(frame, mask, (x1, y1, x2, y2))

            detections.append(
                DetectionInfo(
                    object_id=f"obj-{index + 1:02d}",
                    class_name=class_name,
                    color_name=color_name,
                    color_code=color_code,
                    confidence=confidence,
                    area_ratio=area_ratio,
                    center_x=center_x,
                    center_y=center_y,
                    norm_u=round(center_x / max(1, width), 4),
                    norm_v=round(center_y / max(1, height), 4),
                    bbox=(x1, y1, x2, y2),
                    bgr=bgr,
                )
            )

    detections.sort(key=lambda item: (item.area_ratio, item.confidence), reverse=True)
    output = render_overlay(frame, detections, source_tag, device_label, latency_ms)
    return {"frame": output, "detections": detections, "latency_ms": latency_ms}


class SerialPortManager:
    def __init__(self) -> None:
        self.connection: serial.Serial | None = None
        self.port_name = ""
        self.baudrate = 115200

    @property
    def is_connected(self) -> bool:
        return self.connection is not None and self.connection.is_open

    def available_ports(self) -> list[str]:
        ports = [item.device for item in list_ports.comports()]
        return sorted(ports)

    def connect(self, port_name: str, baudrate: int) -> None:
        self.disconnect()
        self.connection = serial.Serial(
            port=port_name,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2,
            write_timeout=0.2,
        )
        self.port_name = port_name
        self.baudrate = baudrate

    def disconnect(self) -> None:
        if self.connection is not None and self.connection.is_open:
            self.connection.close()
        self.connection = None
    
    def send_line(self, data: Union[str, bytes]) -> None:
        if isinstance(data, str):
            payload = data.encode('utf-8')
        else:
            payload = data
        self.connection.write(payload)
        self.connection.flush()



class InferenceWorker(QObject):
    result_ready = Signal(object)
    log_ready = Signal(str)
    error_ready = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.device_arg, self.device_label = auto_device()

    @Slot(object)
    def process(self, request: dict[str, Any]) -> None:
        frame: np.ndarray = request["frame"]
        model_path = Path(request["model_path"])
        conf = float(request["conf"])
        iou = float(request["iou"])
        source_tag = str(request["source_tag"])
        source_name = str(request["source_name"])

        try:
            if self.model is None or self.model_path != model_path:
                self.log_ready.emit(f"加载模型: {rel_path(model_path)}")
                self.model = YOLO(str(model_path))
                self.model_path = model_path

            started = time.perf_counter()
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                device=self.device_arg,
                imgsz=640,
                retina_masks=True,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            payload = analyze_result(frame, results[0], source_tag, self.device_label, latency_ms)
            payload["device_label"] = self.device_label
            payload["source_name"] = source_name
            self.result_ready.emit(payload)
        except Exception as exc:
            self.error_ready.emit(f"推理失败: {exc}")


class VisionWindow(QMainWindow):
    infer_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.models = discover_models()
        self.serial_manager = SerialPortManager()
        self.current_device_label = auto_device()[1]
        self.current_source_name = "空闲"
        self.current_source_tag = "IDLE"
        self.current_frame: np.ndarray | None = None
        self.current_result_frame: np.ndarray | None = None
        self.current_source_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.last_detections: list[DetectionInfo] = []
        self.last_latency_ms = 0.0
        self.worker_busy = False
        self.layout_mode = ""

        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._next_stream_frame)
        self.reprocess_timer = QTimer(self)
        self.reprocess_timer.setSingleShot(True)
        self.reprocess_timer.timeout.connect(self._dispatch_inference)

        self.worker_thread = QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.worker_thread)
        self.infer_requested.connect(self.worker.process)
        self.worker.result_ready.connect(self._handle_inference_result)
        self.worker.log_ready.connect(lambda message: self._append_log("SYS", message))
        self.worker.error_ready.connect(self._handle_worker_error)
        self.worker_thread.start()

        self._build_ui()
        self._apply_style()
        self._refresh_model_items()
        self._refresh_ports(initial=True)
        self._load_demo_image()
        self._apply_responsive_layout()
        self._update_summary()

    def _build_ui(self) -> None:
        self.setWindowTitle("智能分拣视觉中控台")
        self.resize(1440, 860)
        self.setMinimumSize(1080, 680)

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(8, 8, 8, 6)
        main_layout.setSpacing(8)

        header = QFrame()
        header.setObjectName("Header")
        header.setMinimumHeight(42)
        header.setMaximumHeight(48)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(10)

        self.title_label = QLabel("智能分拣视觉中控台")
        self.title_label.setObjectName("Title")
        self.summary_label = QLabel("等待输入源")
        self.summary_label.setObjectName("Summary")
        self.summary_label.setWordWrap(False)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.summary_label, 1)

        badge_layout = QHBoxLayout()
        badge_layout.setSpacing(6)
        self.device_badge = QLabel()
        self.mode_badge = QLabel()
        badge_layout.addWidget(self.device_badge)
        badge_layout.addWidget(self.mode_badge)
        header_layout.addLayout(badge_layout)
        main_layout.addWidget(header)

        self.content_splitter = QSplitter(Qt.Vertical)
        self.content_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.content_splitter, 1)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        workspace_layout.addWidget(self.main_splitter)

        preview_panel, preview_body = self._make_panel("实时预览", show_title=False)
        preview_layout = QVBoxLayout(preview_body)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.preview_label = QLabel("加载图片、视频或摄像头后，这里会显示带识别框的实时画面。")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setObjectName("Preview")
        self.preview_label.setMinimumSize(680, 460)
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label, 1)
        self.main_splitter.addWidget(preview_panel)

        self.side_panel = QWidget()
        side_panel_layout = QVBoxLayout(self.side_panel)
        side_panel_layout.setContentsMargins(0, 0, 0, 0)
        side_panel_layout.setSpacing(0)

        control_scroll = QScrollArea()
        control_scroll.setObjectName("ControlScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        side_panel_layout.addWidget(control_scroll)

        control_body = QWidget()
        control_layout = QVBoxLayout(control_body)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)

        vision_panel, vision_body = self._make_panel("视觉输入与推理")
        vision_layout = QGridLayout(vision_body)
        vision_layout.setContentsMargins(0, 0, 0, 0)
        vision_layout.setHorizontalSpacing(6)
        vision_layout.setVerticalSpacing(8)

        self.model_combo = QComboBox()
        self.refresh_model_btn = QPushButton("刷新模型")
        self.refresh_model_btn.setProperty("accent", "ghost")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)
        self.conf_value = QLabel("0.35")
        self.conf_value.setObjectName("ValueBadge")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 90)
        self.iou_slider.setValue(45)
        self.iou_value = QLabel("0.45")
        self.iou_value.setObjectName("ValueBadge")

        vision_layout.addWidget(QLabel("模型"), 0, 0)
        vision_layout.addWidget(self.model_combo, 0, 1)
        vision_layout.addWidget(self.refresh_model_btn, 0, 2)
        vision_layout.addWidget(QLabel("置信度"), 1, 0)
        vision_layout.addWidget(self.conf_slider, 1, 1)
        vision_layout.addWidget(self.conf_value, 1, 2)
        vision_layout.addWidget(QLabel("IoU"), 2, 0)
        vision_layout.addWidget(self.iou_slider, 2, 1)
        vision_layout.addWidget(self.iou_value, 2, 2)

        input_buttons = QGridLayout()
        input_buttons.setHorizontalSpacing(6)
        input_buttons.setVerticalSpacing(6)
        self.open_image_btn = QPushButton("加载图片")
        self.open_image_btn.setProperty("accent", "primary")
        self.open_video_btn = QPushButton("加载视频")
        self.open_video_btn.setProperty("accent", "secondary")
        self.open_camera_btn = QPushButton("打开摄像头")
        self.open_camera_btn.setProperty("accent", "secondary")
        self.stop_btn = QPushButton("停止输入")
        self.stop_btn.setProperty("accent", "danger")
        self.save_btn = QPushButton("保存画面")
        self.save_btn.setProperty("accent", "ghost")
        input_buttons.addWidget(self.open_image_btn, 0, 0)
        input_buttons.addWidget(self.open_video_btn, 0, 1)
        input_buttons.addWidget(self.open_camera_btn, 0, 2)
        input_buttons.addWidget(self.stop_btn, 1, 0)
        input_buttons.addWidget(self.save_btn, 1, 1, 1, 2)
        vision_layout.addLayout(input_buttons, 3, 0, 1, 3)
        control_layout.addWidget(vision_panel)

        serial_panel, serial_body = self._make_panel("串口发送")
        serial_layout = QGridLayout(serial_body)
        serial_layout.setContentsMargins(0, 0, 0, 0)
        serial_layout.setHorizontalSpacing(6)
        serial_layout.setVerticalSpacing(8)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["模拟", "串口"])
        self.port_combo = QComboBox()
        self.refresh_port_btn = QPushButton("刷新串口")
        self.refresh_port_btn.setProperty("accent", "ghost")
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "57600", "115200", "230400"])
        self.baud_combo.setCurrentText("115200")
        self.connect_btn = QPushButton("连接串口")
        self.connect_btn.setProperty("accent", "secondary")
        self.disconnect_btn = QPushButton("断开连接")
        self.disconnect_btn.setProperty("accent", "danger")
        self.send_all_btn = QPushButton("发送全部目标")
        self.send_all_btn.setProperty("accent", "warm")

        serial_layout.addWidget(QLabel("模式"), 0, 0)
        serial_layout.addWidget(self.mode_combo, 0, 1, 1, 2)
        serial_layout.addWidget(QLabel("串口"), 1, 0)
        serial_layout.addWidget(self.port_combo, 1, 1)
        serial_layout.addWidget(self.refresh_port_btn, 1, 2)
        serial_layout.addWidget(QLabel("波特率"), 2, 0)
        serial_layout.addWidget(self.baud_combo, 2, 1)
        serial_layout.addWidget(self.connect_btn, 2, 2)
        serial_layout.addWidget(self.disconnect_btn, 3, 1)
        serial_layout.addWidget(self.send_all_btn, 3, 2)
        control_layout.addWidget(serial_panel)
        control_layout.addStretch(1)

        control_scroll.setWidget(control_body)
        self.main_splitter.addWidget(self.side_panel)
        self.main_splitter.setStretchFactor(0, 6)
        self.main_splitter.setStretchFactor(1, 1)

        result_panel, result_body = self._make_panel("识别结果与发送日志", show_title=False)
        result_layout = QVBoxLayout(result_body)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(6)

        tabs = QTabWidget()
        tabs.setObjectName("Tabs")
        tabs.tabBar().setExpanding(False)

        self.target_table = QTableWidget(0, 9)
        self._configure_table(self.target_table, ["序号", "颜色", "代码", "置信度", "x", "y", "u", "v", "发送状态"])
        tabs.addTab(self.target_table, "目标列表")

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setObjectName("Log")
        self.log_edit.setPlaceholderText("串口发送和系统事件会显示在这里。")
        tabs.addTab(self.log_edit, "串口日志")
        tabs.setDocumentMode(True)

        result_layout.addWidget(tabs)
        self.content_splitter.addWidget(workspace)
        self.content_splitter.addWidget(result_panel)
        self.content_splitter.setStretchFactor(0, 7)
        self.content_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([1260, 260])
        self.content_splitter.setSizes([720, 220])

        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("已启动")

        self.refresh_model_btn.clicked.connect(self._refresh_model_items)
        self.conf_slider.valueChanged.connect(lambda value: self.conf_value.setText(f"{value / 100:.2f}"))
        self.iou_slider.valueChanged.connect(lambda value: self.iou_value.setText(f"{value / 100:.2f}"))
        self.conf_slider.valueChanged.connect(self._schedule_reprocess)
        self.iou_slider.valueChanged.connect(self._schedule_reprocess)
        self.open_image_btn.clicked.connect(self._open_image)
        self.open_video_btn.clicked.connect(self._open_video)
        self.open_camera_btn.clicked.connect(self._open_camera)
        self.stop_btn.clicked.connect(self._stop_source)
        self.save_btn.clicked.connect(self._save_current_frame)
        self.model_combo.currentIndexChanged.connect(self._schedule_reprocess)
        self.mode_combo.currentIndexChanged.connect(self._handle_mode_changed)
        self.refresh_port_btn.clicked.connect(self._refresh_ports)
        self.connect_btn.clicked.connect(self._connect_serial)
        self.disconnect_btn.clicked.connect(self._disconnect_serial)
        self.send_all_btn.clicked.connect(self._send_all_targets)

        self._set_badge(self.device_badge, "CPU", "slate")
        self._set_badge(self.mode_badge, "模拟", "warm")

    def _make_panel(self, title: str, show_title: bool = True) -> tuple[QFrame, QWidget]:
        frame = QFrame()
        frame.setObjectName("Panel")
        layout = QVBoxLayout(frame)
        if show_title:
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
        else:
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
        body = QWidget()
        if show_title:
            label = QLabel(title)
            label.setObjectName("PanelTitle")
            layout.addWidget(label)
        layout.addWidget(body, 1)
        return frame, body

    def _set_badge(self, label: QLabel, text: str, tone: str = "neutral") -> None:
        palette = {
            "slate": ("#eaf1f7", "#17324a", "#b8ccde"),
            "teal": ("#e2f5f1", "#0f655a", "#9bcfc5"),
            "warm": ("#fff1e6", "#9a5214", "#efbc8c"),
            "green": ("#e7f6ed", "#266a44", "#b0d8bd"),
            "danger": ("#fff0f0", "#9c2f2f", "#e3b3b3"),
            "neutral": ("#f4f7fa", "#5d7186", "#d6dde6"),
        }
        background, foreground, border = palette.get(tone, palette["neutral"])
        label.setText(text)
        label.setStyleSheet(
            f"background: {background}; color: {foreground}; border: 1px solid {border}; "
            "border-radius: 999px; padding: 4px 10px; font-size: 10px; font-weight: 700;"
        )

    def _configure_table(self, table: QTableWidget, headers: list[str]) -> None:
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        table.horizontalHeader().setSectionResizeMode(0, HEADER_RESIZE)
        table.horizontalHeader().setSectionResizeMode(1, HEADER_RESIZE)
        table.horizontalHeader().setSectionResizeMode(2, HEADER_RESIZE)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.setEditTriggers(NO_EDIT)
        table.setSelectionBehavior(SELECT_ROWS)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#Root {
                background: #edf2f7;
                color: #213547;
                font-family: "HarmonyOS Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei";
                font-size: 11px;
            }
            QFrame#Header {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #fdfefe,
                    stop: 0.55 #f4f7fb,
                    stop: 1 #ecf2f8
                );
                border: 1px solid #d4dde8;
                border-radius: 16px;
            }
            QFrame#Panel, QFrame#InfoStrip {
                background: #fbfdff;
                border: 1px solid #d7e0ea;
                border-radius: 12px;
            }
            QFrame#InfoStrip {
                background: #f4f8fc;
                border-radius: 10px;
            }
            QLabel#Title {
                color: #102235;
                font-size: 15px;
                font-weight: 800;
            }
            QLabel#Summary {
                color: #5f7183;
                font-size: 10px;
            }
            QLabel#PanelTitle {
                color: #12324b;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#Preview {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #fefefe,
                    stop: 0.62 #f2f6fb,
                    stop: 1 #e8eef5
                );
                border: 1px solid #d5dee8;
                border-radius: 16px;
                color: #6a7e91;
                font-size: 12px;
                font-weight: 600;
                padding: 2px;
            }
            QLabel#InfoStripText {
                color: #5c7286;
                font-size: 10px;
            }
            QLabel#ValueBadge {
                background: #eef4f9;
                color: #1e3b58;
                border: 1px solid #d7e0ea;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 700;
                padding: 4px 8px;
                min-width: 50px;
            }
            QScrollArea#ControlScroll, QScrollArea#ControlScroll > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QComboBox, QPlainTextEdit, QTableWidget, QTabWidget::pane {
                background: #ffffff;
                border: 1px solid #d7e0ea;
                border-radius: 10px;
                color: #213547;
            }
            QComboBox {
                min-height: 30px;
                padding: 0 10px;
            }
            QComboBox::drop-down {
                width: 22px;
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                selection-background-color: #e8f2fb;
                selection-color: #102235;
                border: 1px solid #d7e0ea;
            }
            QPushButton {
                min-height: 30px;
                padding: 0 10px;
                border-radius: 8px;
                font-weight: 700;
                border: 1px solid #d7e0ea;
                background: #f6f8fb;
                color: #26405b;
            }
            QPushButton:hover {
                border-color: #b7c6d7;
                background: #eef3f7;
            }
            QPushButton:pressed {
                background: #e5ecf3;
            }
            QPushButton[accent="primary"] {
                background: #19344d;
                color: #ffffff;
                border: 1px solid #19344d;
            }
            QPushButton[accent="primary"]:hover {
                background: #244564;
                border-color: #244564;
            }
            QPushButton[accent="secondary"] {
                background: #e4f5f2;
                color: #0f6258;
                border: 1px solid #9fcfc7;
            }
            QPushButton[accent="secondary"]:hover {
                background: #d8efe9;
                border-color: #83bfb5;
            }
            QPushButton[accent="warm"] {
                background: #fff2e6;
                color: #9b5213;
                border: 1px solid #efbf92;
            }
            QPushButton[accent="warm"]:hover {
                background: #ffe8d4;
                border-color: #e8ad73;
            }
            QPushButton[accent="danger"] {
                background: #fff0f0;
                color: #9c2f2f;
                border: 1px solid #e3b3b3;
            }
            QPushButton[accent="danger"]:hover {
                background: #ffe4e4;
                border-color: #d89797;
            }
            QPushButton[accent="ghost"] {
                background: #f6f8fb;
                color: #35516d;
                border: 1px solid #d4dde8;
            }
            QPushButton:disabled {
                color: #97a7b7;
                background: #f1f4f7;
                border: 1px solid #dde4eb;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #dde7f0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #17324a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #ff9a4d;
                border: 2px solid #ffffff;
                border-radius: 7px;
            }
            QHeaderView::section {
                background: #f2f6fa;
                color: #44586d;
                padding: 6px 5px;
                border: none;
                border-bottom: 1px solid #d7e0ea;
                font-weight: 700;
            }
            QTableWidget {
                alternate-background-color: #f8fbfd;
                selection-background-color: #deebf7;
                selection-color: #102235;
                gridline-color: #ecf1f5;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QPlainTextEdit#Log {
                color: #eaf3fb;
                background: #0f2233;
                border: 1px solid #244761;
                selection-background-color: #2a5878;
                font-family: "JetBrains Mono", "SF Mono", "Consolas";
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabWidget::pane {
                margin-top: 4px;
            }
            QTabBar::tab {
                background: #f2f6fa;
                color: #52667a;
                padding: 6px 12px;
                border: 1px solid #d7e0ea;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #12273b;
            }
            QSplitter::handle {
                background: #e2e8ef;
            }
            QSplitter::handle:horizontal {
                width: 8px;
                margin: 8px 0;
                border-radius: 4px;
            }
            QSplitter::handle:vertical {
                height: 8px;
                margin: 0 8px;
                border-radius: 4px;
            }
            QStatusBar {
                background: #ffffff;
                color: #5d7186;
                border: 1px solid #d7e0ea;
                border-radius: 8px;
                padding: 2px 8px;
            }
            """
        )

    def _selected_model_path(self) -> Path | None:
        index = self.model_combo.currentIndex()
        if index < 0 or index >= len(self.models):
            return None
        return self.models[index]

    def _refresh_model_items(self, *_args) -> None:
        self.models = discover_models()
        current_text = self.model_combo.currentText()
        self.model_combo.clear()
        for path in self.models:
            self.model_combo.addItem(rel_path(path))
        if current_text:
            index = self.model_combo.findText(current_text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        if self.models and self.model_combo.currentIndex() < 0:
            self.model_combo.setCurrentIndex(0)
        self._update_summary()

    def _refresh_ports(self, *_args, initial: bool = False) -> None:
        ports = self.serial_manager.available_ports()
        current_text = self.port_combo.currentText()
        self.port_combo.clear()
        if ports:
            self.port_combo.addItems(ports)
            if current_text:
                index = self.port_combo.findText(current_text)
                if index >= 0:
                    self.port_combo.setCurrentIndex(index)
        else:
            self.port_combo.addItem("未发现串口")

        if initial and ports:
            self.mode_combo.setCurrentText("串口")
        elif initial:
            self.mode_combo.setCurrentText("模拟")

        self._update_serial_controls()
        self._update_summary()

    def _handle_mode_changed(self, *_args) -> None:
        if self.mode_combo.currentText() == "模拟":
            self._disconnect_serial(silent=True)
        self._update_serial_controls()
        self._update_summary()

    def _update_serial_controls(self) -> None:
        serial_mode = self.mode_combo.currentText() == "串口"
        has_real_port = self.port_combo.count() > 0 and self.port_combo.currentText() != "未发现串口"
        self.port_combo.setEnabled(serial_mode)
        self.refresh_port_btn.setEnabled(serial_mode)
        self.baud_combo.setEnabled(serial_mode)
        self.connect_btn.setEnabled(serial_mode and has_real_port and not self.serial_manager.is_connected)
        self.disconnect_btn.setEnabled(serial_mode and self.serial_manager.is_connected)

    def _build_serial_payload(self, detection: DetectionInfo) -> dict[str, Any]:
        return {
            "x": detection.center_x,
            "y": detection.center_y,
            "u": round(detection.norm_u, 4),
            "v": round(detection.norm_v, 4),
            "color": detection.color_name,
            "color_code": detection.color_code,
        }

    def _append_log(self, tag: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{timestamp}] {tag} {message}")

    def _show_frame(self, frame: np.ndarray) -> None:
        pixmap = frame_to_pixmap(frame).scaled(self.preview_label.size(), KEEP_ASPECT, SMOOTH_TRANSFORM)
        self.preview_label.setPixmap(pixmap)

    def _load_demo_image(self) -> None:
        demo_path = discover_demo_image()
        if demo_path is None:
            return
        frame = read_local_image(demo_path)
        if frame is None:
            return
        self.current_source_path = demo_path
        self.current_source_name = f"本地图像 · {demo_path.name}"
        self.current_source_tag = "IMAGE"
        self.current_frame = frame
        self._show_frame(frame)
        self._update_summary()
        self._dispatch_inference()

    def _dispatch_inference(self) -> None:
        if self.worker_busy or self.current_frame is None:
            return
        model_path = self._selected_model_path()
        if model_path is None:
            self._append_log("SYS", "未找到可用模型")
            return

        self.worker_busy = True
        request = {
            "frame": self.current_frame.copy(),
            "model_path": str(model_path),
            "conf": self.conf_slider.value() / 100.0,
            "iou": self.iou_slider.value() / 100.0,
            "source_tag": self.current_source_tag,
            "source_name": self.current_source_name,
        }
        self.infer_requested.emit(request)
        self.status_bar.showMessage("推理中…", 1500)

    def _schedule_reprocess(self, *_args) -> None:
        if self.current_frame is None or self.stream_timer.isActive():
            return
        self.reprocess_timer.start(180)

    def _fill_target_table(self) -> None:
        self.target_table.setRowCount(len(self.last_detections))
        for row, detection in enumerate(self.last_detections, start=1):
            values = [
                row,
                detection.color_name,
                detection.color_code,
                f"{detection.confidence:.2f}",
                detection.center_x,
                detection.center_y,
                f"{detection.norm_u:.4f}",
                f"{detection.norm_v:.4f}",
                detection.send_status,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                if column in {1, 2}:
                    r, g, b = detection.bgr[2], detection.bgr[1], detection.bgr[0]
                    item.setBackground(QColor(r, g, b, 70))
                if column == 8:
                    if detection.send_status == "已发送":
                        item.setBackground(QColor(50, 142, 87, 110))
                    elif detection.send_status == "发送失败":
                        item.setBackground(QColor(190, 72, 72, 110))
                self.target_table.setItem(row - 1, column, item)

    def _update_summary(self) -> None:
        source_name = self.current_source_name or "空闲"
        if self.mode_combo.currentText() == "模拟":
            serial_state = "模拟"
        elif self.serial_manager.is_connected:
            serial_state = self.serial_manager.port_name
        else:
            serial_state = "未连接"

        frame = self.current_result_frame if self.current_result_frame is not None else self.current_frame
        if frame is not None:
            height, width = frame.shape[:2]
            resolution_text = f"{width} x {height}"
        else:
            resolution_text = "--"

        latency_text = f"{self.last_latency_ms:.1f} ms" if self.last_latency_ms > 0 else "等待推理"
        target_count = len(self.last_detections)

        if self.current_source_tag == "IDLE":
            self.summary_label.setText("等待输入源")
        else:
            self.summary_label.setText(f"{short_text(source_name, 18)}  ·  {resolution_text}  ·  {target_count} 个目标  ·  {latency_text}")

        device_tone = "teal" if self.current_device_label in {"CUDA", "MPS"} else "slate"
        if self.mode_combo.currentText() == "模拟":
            mode_text = "模拟"
            mode_tone = "warm"
        elif self.serial_manager.is_connected:
            mode_text = short_text(serial_state, 10)
            mode_tone = "green"
        else:
            mode_text = "未连接"
            mode_tone = "danger"

        self._set_badge(self.device_badge, self.current_device_label, device_tone)
        self._set_badge(self.mode_badge, mode_text, mode_tone)

    def _connect_serial(self, *_args) -> None:
        if self.mode_combo.currentText() != "串口":
            QMessageBox.information(self, "当前为模拟模式", "请先将模式切换为串口。")
            return
        port_name = self.port_combo.currentText()
        if not port_name or port_name == "未发现串口":
            QMessageBox.warning(self, "没有可用串口", "请先刷新串口并选择有效端口。")
            return
        try:
            self.serial_manager.connect(port_name, int(self.baud_combo.currentText()))
            self._append_log("SYS", f"串口已连接: {port_name}@{self.baud_combo.currentText()}")
            self.status_bar.showMessage("串口已连接", 2000)
        except Exception as exc:
            QMessageBox.warning(self, "串口连接失败", str(exc))
            self._append_log("ERR", f"串口连接失败: {exc}")
        self._update_serial_controls()
        self._update_summary()

    def _disconnect_serial(self, *_args, silent: bool = False) -> None:
        if self.serial_manager.is_connected:
            self.serial_manager.disconnect()
            if not silent:
                self._append_log("SYS", "串口已断开")
                self.status_bar.showMessage("串口已断开", 2000)
        self._update_serial_controls()
        self._update_summary()

    def _send_all_targets(self, *_args) -> None:
        """核心发送逻辑：转换坐标 -> 姿态解算 -> 发送角度"""
        if not self.last_detections:
            QMessageBox.information(self, "信息", "当前没有识别到任何目标。")
            return

        serial_mode = self.mode_combo.currentText()
        if serial_mode == "串口" and not self.serial_manager.is_connected:
            QMessageBox.warning(self, "串口未连接", "请先连接串口。")
            return

        for detection in self.last_detections:
            
            color_byte = ord(COLOR_CODE_TO_CHAR.get(detection.color_code, 'A'))

         
            PIXELS_PER_CM = 15.65
            
   
            REF_X = 427.0       
            REF_Y = 230.0        
            OFFSET_RIGHT = 13.5  
            OFFSET_UP = 0     
        

          
            base_x_px = REF_X + (OFFSET_RIGHT * PIXELS_PER_CM)
            base_y_px = REF_Y - (OFFSET_UP * PIXELS_PER_CM) 
            
            cx = detection.center_x
            cy = detection.center_y
            
           
            arm_x_cm = (base_y_px - cy) / PIXELS_PER_CM
            arm_y_cm = (base_x_px - cx) / PIXELS_PER_CM

         
            j1, j2, j3, j4 = calculate_joints(arm_x_cm, arm_y_cm)
            
            if j1 is None:
                self._append_log("WARN", f"目标({arm_x_cm:.1f}, {arm_y_cm:.1f})cm 超出机械臂工作空间")
                detection.send_status = "不可达"
                continue

            
            servo_angles = map_to_servos(j1, j2, j3, j4)
            angles_payload = [max(0, min(255, int(round(a)))) for a in servo_angles]

            
            try:
               
                packet = struct.pack('<BBB6BB', 0x2C, 0x12, color_byte, *angles_payload, 0x5B)
                
                if serial_mode == "串口":
                    self.serial_manager.send_line(packet)
                    self._append_log("TX", f"发送角度: {angles_payload}")
                else:
                    self._append_log("SIM", f"模拟发送角度: {angles_payload}")
                    self._append_log("WARN", f"目标({arm_x_cm:.1f}, {arm_y_cm:.1f})cm 超出机械臂工作空间")
                
                detection.send_status = "已发送"
            except Exception as e:
                self._append_log("ERR", f"发送异常: {e}")
                detection.send_status = "失败"

            self._fill_target_table()
            self.status_bar.showMessage("指令下发完成", 2000)
        
    def _open_image(self, *_args) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            str(ROOT / "datasets/images" if (ROOT / "datasets/images").exists() else ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not file_path:
            return
        self._stop_source(clear_source=False)
        path = Path(file_path)
        frame = read_local_image(path)
        if frame is None:
            QMessageBox.warning(self, "打开失败", "图片读取失败，请确认文件格式是否正确。")
            return
        self.current_source_path = path
        self.current_source_name = f"本地图像 · {path.name}"
        self.current_source_tag = "IMAGE"
        self.current_frame = frame
        self._show_frame(frame)
        self._dispatch_inference()
        self._update_summary()

    def _open_video(self, *_args) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开视频",
            str(ROOT),
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if not file_path:
            return
        self._stop_source(clear_source=False)
        path = Path(file_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            QMessageBox.warning(self, "打开失败", "视频打开失败，请确认文件可正常播放。")
            return
        self.capture = capture
        self.current_source_path = path
        self.current_source_name = f"本地视频 · {path.name}"
        self.current_source_tag = "VIDEO"
        self.stream_timer.start(30)
        self._append_log("SYS", f"开始播放视频: {path.name}")
        self._update_summary()

    def _open_camera(self, *_args) -> None:
        self._stop_source(clear_source=False)
        for index in range(3):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                self.capture = capture
                self.current_source_path = None
                self.current_source_name = f"摄像头 #{index}"
                self.current_source_tag = "CAMERA"
                self.stream_timer.start(30)
                self._append_log("SYS", f"已连接摄像头: {index}")
                self._update_summary()
                return
        QMessageBox.warning(self, "摄像头不可用", "未检测到可用摄像头。")

    def _next_stream_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self._append_log("SYS", "输入流结束，已停止播放。")
            self._stop_source(clear_source=False)
            return
        self.current_frame = frame
        self._show_frame(frame)
        self._dispatch_inference()

    def _stop_source(self, *_args, clear_source: bool = True) -> None:
        self.stream_timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if clear_source:
            self.current_source_name = "空闲"
            self.current_source_tag = "IDLE"
            self.current_source_path = None
        self._update_summary()

    def _save_current_frame(self, *_args) -> None:
        frame = self.current_result_frame if self.current_result_frame is not None else self.current_frame
        if frame is None:
            QMessageBox.information(self, "没有画面", "当前没有可保存的画面。")
            return
        output_dir = ROOT / "runs"
        output_dir.mkdir(exist_ok=True)
        filename = time.strftime("result_%Y%m%d_%H%M%S.jpg")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存画面",
            str(output_dir / filename),
            "Images (*.jpg *.png)",
        )
        if not file_path:
            return
        if write_local_image(Path(file_path), frame):
            self._append_log("SYS", f"已保存画面: {Path(file_path).name}")
            self.status_bar.showMessage("画面已保存", 2000)
        else:
            QMessageBox.warning(self, "保存失败", "画面保存失败。")

    def _handle_inference_result(self, payload: dict[str, Any]) -> None:
        self.worker_busy = False
        self.current_device_label = payload["device_label"]
        self.current_result_frame = payload["frame"]
        self.last_detections = payload["detections"]
        self.last_latency_ms = float(payload["latency_ms"])
        for detection in self.last_detections:
            detection.send_status = "待发送"
        self._fill_target_table()
        self._show_frame(self.current_result_frame)
        self.status_bar.showMessage(f"识别到 {len(self.last_detections)} 个目标", 1500)
        self._update_summary()

    def _handle_worker_error(self, message: str) -> None:
        self.worker_busy = False
        self.last_latency_ms = 0.0
        self._append_log("ERR", message)
        self.status_bar.showMessage(message, 2500)
        self._update_summary()

    def _apply_responsive_layout(self) -> None:
        compact = self.width() < 1360
        new_mode = "compact" if compact else "wide"
        if new_mode == self.layout_mode:
            return
        self.layout_mode = new_mode
        if compact:
            self.side_panel.setMinimumWidth(0)
            self.side_panel.setMaximumWidth(16777215)
            self.main_splitter.setOrientation(Qt.Vertical)
            self.main_splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
            self.content_splitter.setSizes([int(self.height() * 0.56), int(self.height() * 0.44)])
        else:
            self.side_panel.setMinimumWidth(320)
            self.side_panel.setMaximumWidth(360)
            self.main_splitter.setOrientation(Qt.Horizontal)
            self.main_splitter.setSizes([int(self.width() * 0.84), int(self.width() * 0.16)])
            self.content_splitter.setSizes([int(self.height() * 0.68), int(self.height() * 0.32)])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()
        if self.current_result_frame is not None:
            self._show_frame(self.current_result_frame)
        elif self.current_frame is not None:
            self._show_frame(self.current_frame)

    def closeEvent(self, event) -> None:
        self._stop_source(clear_source=False)
        self._disconnect_serial(silent=True)
        self.worker_thread.quit()
        self.worker_thread.wait(1500)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = VisionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()