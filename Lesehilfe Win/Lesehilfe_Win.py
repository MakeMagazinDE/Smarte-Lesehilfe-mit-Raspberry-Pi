#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
Projekt:   Lesehilfe für Senioren (Windows-Version)
Datei:     Lesehilfe_Windows.py

Ersteller: Benjamin Deiss
Version:   1.6.1-win
Datum:     22.01.2026

Beschreibung:
Echtzeit-Kamera-Vorschau mit Zoom, Kontrast- und
Helligkeitssteuerung, OCR-Texterkennung mit Tesseract,
Sprachausgabe mit Google TTS (online) und pyttsx3 (offline).
Angepasst für Windows mit USB-Kamera und Menüleiste.
============================================================
"""

from __future__ import annotations

import cv2
import numpy as np
import json
import os
import sys
import subprocess
import time
import pytesseract
import threading
import requests
from gtts import gTTS
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Callable, Any
from enum import IntEnum
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Windows TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Pygame für Audio-Wiedergabe auf Windows
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# KONSTANTEN & ENUMS
# =============================================================================

class QualityMode(IntEnum):
    LINEAR = 0
    LANCZOS = 1
    SHARPEN_LIGHT = 2
    SHARPEN_STRONG = 3


# Dateipfade für Windows
TEMP_DIR = Path(os.environ.get('TEMP', 'C:/Temp'))
TEMP_AUDIO_FILE = TEMP_DIR / "lesehilfe_speak.mp3"
SETTINGS_FILE = Path("settings.json")

# Tesseract-Pfad für Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Display-Konstanten
DEFAULT_RESOLUTION = (1280, 1024)
OUTPUT_RESOLUTION: Tuple[int, int] = DEFAULT_RESOLUTION
TARGET_ASPECT_RATIO: float = 4 / 3

# Zoom-Grenzen
ZOOM_CONFIG = (1.0, 8.0, 0.5)

# Bildverarbeitungs-Konstanten
BRIGHTNESS_TARGET = 127
BRIGHTNESS_STEP = 12.7
BRIGHTNESS_MAX = 75
CONTRAST_LOW_THRESHOLD = 40
CONTRAST_HIGH_VALUE = 1.4
HISTORY_SIZE = 5

# OCR-Konstanten
OCR_SIZE = (800, 600)
OCR_CONFIDENCE_THRESHOLD = 60
OCR_OVERLAY_ALPHA = 0.8

# Globale Variablen
camera: cv2.VideoCapture
settings: 'Settings'
state: 'ApplicationState'


# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

@dataclass
class Settings:
    zoom_factor: float = 1.0
    rotation: int = 0
    quality_mode: int = QualityMode.SHARPEN_STRONG
    contrast_alpha: float = 1.2
    brightness_beta: float = 0.0
    auto_brightness: bool = True
    color_order: str = "RGB"
    camera_resolution: Tuple[int, int] = (1920, 1080)
    camera_index: int = 0
    monochrome_mode: bool = False
    kalibrierungs_punkte: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    tts_google_lang: str = "de"
    tts_google_slow: bool = False
    tts_pyttsx_rate: int = 150
    tts_pyttsx_volume: float = 1.0
    
    def validate(self) -> None:
        if not (ZOOM_CONFIG[0] <= self.zoom_factor <= ZOOM_CONFIG[1]):
            raise ValueError(f"Zoom muss zwischen {ZOOM_CONFIG[0]} und {ZOOM_CONFIG[1]} liegen")


@dataclass
class ApplicationState:
    running: bool = True
    changed: bool = False
    ocr_active: bool = False
    invert_contrast: bool = False
    status_text: str = ""
    status_end_time: float = 0
    last_recognized_text: str = ""
    last_spoken_text: str = ""
    last_processed_frame: Optional[np.ndarray] = None
    tts_process: Optional[subprocess.Popen] = None
    tts_thread_lock: threading.Lock = field(default_factory=threading.Lock)
    tts_active: bool = False
    brightness_history: list[float] = field(default_factory=list)
    contrast_history: list[float] = field(default_factory=list)
    clicked_points: list[Tuple[int, int]] = field(default_factory=list)
    window_size: Tuple[int, int] = DEFAULT_RESOLUTION


# =============================================================================
# DATEI-MANAGEMENT
# =============================================================================

class SettingsManager:
    @staticmethod
    def load() -> Settings:
        if not SETTINGS_FILE.exists():
            return Settings()
        try:
            with SETTINGS_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("kalibrierungs_punkte") is not None:
                data["kalibrierungs_punkte"] = np.array(data["kalibrierungs_punkte"], dtype=np.float32)
            if data.get("transform_matrix") is not None:
                data["transform_matrix"] = np.array(data["transform_matrix"], dtype=np.float32)
            if data.get("camera_resolution") and isinstance(data["camera_resolution"], list):
                data["camera_resolution"] = tuple(data["camera_resolution"])
            valid_fields = {k: v for k, v in data.items() if k in Settings.__annotations__}
            s = Settings(**valid_fields)
            s.validate()
            return s
        except Exception as e:
            logger.error(f"Fehler beim Laden: {e}")
            return Settings()
    
    @staticmethod
    def save(settings: Settings) -> None:
        try:
            data = asdict(settings)
            if data.get("kalibrierungs_punkte") is not None:
                data["kalibrierungs_punkte"] = data["kalibrierungs_punkte"].tolist()
            if data.get("transform_matrix") is not None:
                data["transform_matrix"] = data["transform_matrix"].tolist()
            if isinstance(data.get("camera_resolution"), tuple):
                data["camera_resolution"] = list(data["camera_resolution"])
            with SETTINGS_FILE.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")


# =============================================================================
# BILDVERARBEITUNG
# =============================================================================

class ImageProcessor:
    def __init__(self, settings: Settings, state: ApplicationState):
        self.settings = settings
        self.state = state
    
    def process(self, frame: np.ndarray, apply_zoom: bool = True, 
                apply_sharpening: bool = True, apply_color_correction: bool = True) -> np.ndarray:
        frame = self._rotate(frame)
        frame = self._crop_to_aspect_ratio(frame)
        if self.settings.transform_matrix is not None:
            frame = self._apply_perspective_warp(frame)
        if apply_zoom and self.settings.zoom_factor > 1.0:
            frame = self._apply_zoom(frame)
        frame = self._scale_to_output(frame)
        if self.settings.monochrome_mode:
            frame = self._convert_to_monochrome(frame)
        if apply_color_correction and not self.settings.monochrome_mode:
            frame = self._adjust_brightness_contrast(frame)
        if apply_sharpening and self.settings.quality_mode >= QualityMode.SHARPEN_LIGHT:
            frame = self._apply_sharpening(frame)
        return frame
    
    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        rotation_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if self.settings.rotation in rotation_map:
            return cv2.rotate(frame, rotation_map[self.settings.rotation])
        return frame
    
    def _crop_to_aspect_ratio(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        current_aspect = w / h
        if current_aspect > TARGET_ASPECT_RATIO:
            new_w = int(h * TARGET_ASPECT_RATIO)
            x1 = (w - new_w) // 2
            return frame[:, x1:x1+new_w]
        else:
            new_h = int(w / TARGET_ASPECT_RATIO)
            y1 = (h - new_h) // 2
            return frame[y1:y1+new_h, :]
    
    def _apply_perspective_warp(self, frame: np.ndarray) -> np.ndarray:
        try:
            h, w = frame.shape[:2]
            warp_h = h
            warp_w = int(warp_h * TARGET_ASPECT_RATIO)
            return cv2.warpPerspective(frame, self.settings.transform_matrix, (warp_w, warp_h))
        except:
            return frame
    
    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        zoom_w = int(w / self.settings.zoom_factor)
        zoom_h = int(h / self.settings.zoom_factor)
        x1 = (w - zoom_w) // 2
        y1 = (h - zoom_h) // 2
        return frame[y1:y1+zoom_h, x1:x1+zoom_w]
    
    def _scale_to_output(self, frame: np.ndarray) -> np.ndarray:
        interpolation = cv2.INTER_LANCZOS4 if self.settings.quality_mode > QualityMode.LINEAR else cv2.INTER_LINEAR
        return cv2.resize(frame, self.state.window_size, interpolation=interpolation)
    
    def _convert_to_monochrome(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        if self.settings.auto_brightness:
            auto_alpha, auto_beta = self._calculate_auto_params(frame)
            final_alpha = self.settings.contrast_alpha * auto_alpha
            final_beta = auto_beta + self.settings.brightness_beta
        else:
            final_alpha = self.settings.contrast_alpha
            final_beta = self.settings.brightness_beta
        return cv2.convertScaleAbs(frame, alpha=final_alpha, beta=final_beta)
    
    def _calculate_auto_params(self, frame: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        std_dev = np.std(gray)
        diff = BRIGHTNESS_TARGET - avg_brightness
        raw_beta = np.clip(round(diff / BRIGHTNESS_STEP) * BRIGHTNESS_STEP, -BRIGHTNESS_MAX, BRIGHTNESS_MAX)
        raw_alpha = CONTRAST_HIGH_VALUE if std_dev < CONTRAST_LOW_THRESHOLD else 1.2
        self.state.brightness_history.append(raw_beta)
        self.state.contrast_history.append(raw_alpha)
        if len(self.state.brightness_history) > HISTORY_SIZE:
            self.state.brightness_history.pop(0)
            self.state.contrast_history.pop(0)
        return (sum(self.state.contrast_history) / len(self.state.contrast_history),
                sum(self.state.brightness_history) / len(self.state.brightness_history))
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        sigma = 1.0 if self.settings.quality_mode == QualityMode.SHARPEN_LIGHT else 2.0
        weight = 1.3 if self.settings.quality_mode == QualityMode.SHARPEN_LIGHT else 1.8
        blur = cv2.GaussianBlur(frame, (0, 0), sigma)
        return cv2.addWeighted(frame, weight, blur, -(weight - 1.0), 0)


# =============================================================================
# OCR & TTS
# =============================================================================

class OCRProcessor:
    def __init__(self, state: ApplicationState):
        self.state = state
    
    def recognize_text_fast(self, frame: np.ndarray) -> str:
        ocr_frame = cv2.resize(frame, OCR_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
        try:
            text = pytesseract.image_to_string(gray, lang='deu', config='--psm 3')
            self.state.last_recognized_text = text.strip()
            return text.strip()
        except Exception as e:
            logger.error(f"OCR-Fehler: {e}")
            return ""
    
    def process_with_overlay(self, frame: np.ndarray) -> np.ndarray:
        ocr_frame = cv2.resize(frame, OCR_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
        try:
            data = pytesseract.image_to_data(gray, lang='deu', output_type=pytesseract.Output.DICT, config='--psm 3')
        except:
            self.state.last_recognized_text = ""
            return frame
        
        overlay = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_image)
        recognized_words = []
        scale_x = frame.shape[1] / OCR_SIZE[0]
        scale_y = frame.shape[0] / OCR_SIZE[1]
        bg_color = (0, 0, 0) if self.state.invert_contrast else (255, 255, 255)
        text_color = (255, 255, 255) if self.state.invert_contrast else (0, 0, 0)
        
        font_path = None
        for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
            if Path(fp).exists():
                font_path = fp
                break
        
        for i, conf in enumerate(data['conf']):
            if int(float(conf)) > OCR_CONFIDENCE_THRESHOLD and data['text'][i].strip():
                text = data['text'][i].strip()
                if len(text.replace('?', '').replace('�', '')) == 0:
                    continue
                recognized_words.append(text)
                x, y = int(data['left'][i] * scale_x), int(data['top'][i] * scale_y)
                w, h = int(data['width'][i] * scale_x), int(data['height'][i] * scale_y)
                font_size = max(int(h * 0.9), 16)
                try:
                    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                padding = int(h * 0.15)
                draw.rectangle([max(0, x-padding), max(0, y-padding), 
                               min(frame.shape[1], x+w+padding), min(frame.shape[0], y+h+padding)], fill=bg_color)
                draw.text((x, y), text, font=font, fill=text_color)
        
        overlay = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.state.last_recognized_text = " ".join(recognized_words)
        return cv2.addWeighted(overlay, OCR_OVERLAY_ALPHA, frame, 1 - OCR_OVERLAY_ALPHA, 0)


class TTSManager:
    def __init__(self, settings: Settings, state: ApplicationState):
        self.settings = settings
        self.state = state
        self._pyttsx_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self._pyttsx_engine = pyttsx3.init()
                self._pyttsx_engine.setProperty('rate', self.settings.tts_pyttsx_rate)
                self._pyttsx_engine.setProperty('volume', self.settings.tts_pyttsx_volume)
            except:
                pass
    
    def speak(self, text: str) -> None:
        self._stop_current_playback()
        text_to_speak = text.strip() or "Kein Text erkannt."
        
        if text_to_speak == self.state.last_spoken_text and TEMP_AUDIO_FILE.exists() and PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.load(str(TEMP_AUDIO_FILE))
                pygame.mixer.music.play()
                return
            except:
                pass
        
        if self._check_internet():
            if self._speak_with_gtts(text_to_speak):
                return
        self._speak_with_pyttsx(text_to_speak)
    
    def _stop_current_playback(self) -> None:
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def _check_internet(self, timeout: int = 2) -> bool:
        try:
            requests.get("http://www.google.com", timeout=timeout)
            return True
        except:
            return False
    
    def _speak_with_gtts(self, text: str) -> bool:
        try:
            tts = gTTS(text=text, lang=self.settings.tts_google_lang, slow=self.settings.tts_google_slow)
            tts.save(str(TEMP_AUDIO_FILE))
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(str(TEMP_AUDIO_FILE))
                pygame.mixer.music.play()
                self.state.last_spoken_text = text
                return True
        except:
            pass
        return False
    
    def _speak_with_pyttsx(self, text: str) -> None:
        if not self._pyttsx_engine:
            return
        def speak_thread():
            self._pyttsx_engine.say(text)
            self._pyttsx_engine.runAndWait()
        threading.Thread(target=speak_thread, daemon=True).start()


# =============================================================================
# TKINTER GUI MIT MENÜLEISTE
# =============================================================================

class LesehilfeGUI:
    def __init__(self):
        global settings, state, camera
        
        # Lade Einstellungen
        settings = SettingsManager.load()
        state = ApplicationState()
        state.window_size = DEFAULT_RESOLUTION
        
        # Hauptfenster
        self.root = tk.Tk()
        self.root.title("Lesehilfe für Senioren - Version 1.6.1")
        self.root.geometry(f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}")
        self.root.minsize(640, 480)
        
        # Menüleiste erstellen
        self._create_menu()
        
        # Canvas für Kamera-Bild
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Statusleiste
        self.status_var = tk.StringVar(value="Bereit")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Kamera initialisieren
        camera = cv2.VideoCapture(settings.camera_index)
        if not camera.isOpened():
            messagebox.showerror("Fehler", f"Kamera {settings.camera_index} konnte nicht geöffnet werden!")
            sys.exit(1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_resolution[1])
        
        # Komponenten
        self.image_processor = ImageProcessor(settings, state)
        self.ocr_processor = OCRProcessor(state)
        self.tts_manager = TTSManager(settings, state)
        
        # Tastatur-Bindings
        self.root.bind('<Key>', self._on_key)
        self.root.bind('<Configure>', self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Kamera-Update starten
        self._update_frame()
    
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # === Datei-Menü ===
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Einstellungen speichern", command=self._save_settings)
        file_menu.add_command(label="Einstellungen neu laden", command=self._reload_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self._on_close, accelerator="Q")
        
        # === Ansicht-Menü ===
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ansicht", menu=view_menu)
        
        # Zoom-Untermenü
        zoom_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Zoom", menu=zoom_menu)
        for z in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
            zoom_menu.add_command(label=f"{z:.1f}x", command=lambda z=z: self._set_zoom(z))
        
        # Rotation-Untermenü
        rotation_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Rotation", menu=rotation_menu)
        for r in [0, 90, 180, 270]:
            rotation_menu.add_command(label=f"{r}°", command=lambda r=r: self._set_rotation(r))
        
        # Qualität-Untermenü
        quality_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Qualität", menu=quality_menu)
        quality_names = ["Linear", "Lanczos", "Schärfe leicht", "Schärfe stark"]
        for i, name in enumerate(quality_names):
            quality_menu.add_command(label=name, command=lambda i=i: self._set_quality(i))
        
        view_menu.add_separator()
        
        # Checkbuttons
        self.monochrome_var = tk.BooleanVar(value=settings.monochrome_mode)
        view_menu.add_checkbutton(label="Schwarz-Weiß", variable=self.monochrome_var, 
                                  command=self._toggle_monochrome)
        
        self.invert_var = tk.BooleanVar(value=state.invert_contrast)
        view_menu.add_checkbutton(label="Invertieren", variable=self.invert_var,
                                  command=self._toggle_invert)
        
        self.ocr_var = tk.BooleanVar(value=state.ocr_active)
        view_menu.add_checkbutton(label="OCR-Overlay", variable=self.ocr_var,
                                  command=self._toggle_ocr)
        
        # === Bild-Menü ===
        image_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bild", menu=image_menu)
        
        # Helligkeit-Untermenü
        brightness_menu = tk.Menu(image_menu, tearoff=0)
        image_menu.add_cascade(label="Helligkeit", menu=brightness_menu)
        for b in [-50, -25, -10, 0, 10, 25, 50]:
            brightness_menu.add_command(label=f"{b:+d}", command=lambda b=b: self._set_brightness(b))
        
        # Kontrast-Untermenü
        contrast_menu = tk.Menu(image_menu, tearoff=0)
        image_menu.add_cascade(label="Kontrast", menu=contrast_menu)
        for c in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
            contrast_menu.add_command(label=f"{c:.1f}", command=lambda c=c: self._set_contrast(c))
        
        image_menu.add_separator()
        
        self.auto_brightness_var = tk.BooleanVar(value=settings.auto_brightness)
        image_menu.add_checkbutton(label="Auto-Helligkeit", variable=self.auto_brightness_var,
                                   command=self._toggle_auto_brightness)
        
        image_menu.add_separator()
        image_menu.add_command(label="Kalibrierung...", command=self._start_calibration)
        image_menu.add_command(label="Kalibrierung zurücksetzen", command=self._reset_calibration)
        
        # === Kamera-Menü ===
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Kamera", menu=camera_menu)
        
        # Kamera-Index
        camera_index_menu = tk.Menu(camera_menu, tearoff=0)
        camera_menu.add_cascade(label="Kamera wählen", menu=camera_index_menu)
        for i in range(5):
            camera_index_menu.add_command(label=f"Kamera {i}", command=lambda i=i: self._set_camera(i))
        
        # Auflösung
        resolution_menu = tk.Menu(camera_menu, tearoff=0)
        camera_menu.add_cascade(label="Auflösung", menu=resolution_menu)
        resolutions = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
        for w, h in resolutions:
            resolution_menu.add_command(label=f"{w}x{h}", command=lambda w=w, h=h: self._set_resolution(w, h))
        
        # === Sprache-Menü ===
        speech_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Sprache", menu=speech_menu)
        speech_menu.add_command(label="Text vorlesen", command=self._start_tts, accelerator="Leertaste")
        speech_menu.add_command(label="Wiedergabe stoppen", command=self._stop_tts)
        speech_menu.add_separator()
        
        # TTS-Geschwindigkeit
        tts_speed_menu = tk.Menu(speech_menu, tearoff=0)
        speech_menu.add_cascade(label="Geschwindigkeit", menu=tts_speed_menu)
        for rate in [100, 125, 150, 175, 200]:
            tts_speed_menu.add_command(label=f"{rate} WPM", command=lambda r=rate: self._set_tts_rate(r))
        
        # TTS-Sprache
        tts_lang_menu = tk.Menu(speech_menu, tearoff=0)
        speech_menu.add_cascade(label="Sprache (Google TTS)", menu=tts_lang_menu)
        languages = [("Deutsch", "de"), ("Englisch", "en"), ("Französisch", "fr"), ("Spanisch", "es")]
        for name, code in languages:
            tts_lang_menu.add_command(label=name, command=lambda c=code: self._set_tts_lang(c))
        
        # === Hilfe-Menü ===
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        help_menu.add_command(label="Tastenkürzel", command=self._show_shortcuts)
        help_menu.add_command(label="Über", command=self._show_about)
    
    # === Menü-Handler ===
    
    def _save_settings(self):
        SettingsManager.save(settings)
        self._update_status("Einstellungen gespeichert")
    
    def _reload_settings(self):
        global settings
        settings = SettingsManager.load()
        self.monochrome_var.set(settings.monochrome_mode)
        self.auto_brightness_var.set(settings.auto_brightness)
        self._update_status("Einstellungen neu geladen")
    
    def _set_zoom(self, zoom: float):
        settings.zoom_factor = zoom
        state.changed = True
        self._update_status(f"Zoom: {zoom:.1f}x")
    
    def _set_rotation(self, rotation: int):
        settings.rotation = rotation
        state.changed = True
        self._update_status(f"Rotation: {rotation}°")
    
    def _set_quality(self, mode: int):
        settings.quality_mode = mode
        names = ["Linear", "Lanczos", "Schärfe leicht", "Schärfe stark"]
        state.changed = True
        self._update_status(f"Qualität: {names[mode]}")
    
    def _toggle_monochrome(self):
        settings.monochrome_mode = self.monochrome_var.get()
        state.changed = True
        self._update_status("Schwarz-Weiß " + ("An" if settings.monochrome_mode else "Aus"))
    
    def _toggle_invert(self):
        state.invert_contrast = self.invert_var.get()
        self._update_status("Invertiert " + ("An" if state.invert_contrast else "Aus"))
    
    def _toggle_ocr(self):
        state.ocr_active = self.ocr_var.get()
        self._update_status("OCR-Overlay " + ("An" if state.ocr_active else "Aus"))
    
    def _set_brightness(self, brightness: float):
        settings.brightness_beta = brightness
        state.changed = True
        self._update_status(f"Helligkeit: {brightness:+.0f}")
    
    def _set_contrast(self, contrast: float):
        settings.contrast_alpha = contrast
        state.changed = True
        self._update_status(f"Kontrast: {contrast:.1f}")
    
    def _toggle_auto_brightness(self):
        settings.auto_brightness = self.auto_brightness_var.get()
        state.changed = True
        self._update_status("Auto-Helligkeit " + ("An" if settings.auto_brightness else "Aus"))
    
    def _start_calibration(self):
        messagebox.showinfo("Kalibrierung", 
                          "Kalibrierung wird in separatem Fenster geöffnet.\n\n"
                          "1. Klicken Sie 4 Eckpunkte eines Rechtecks\n"
                          "2. Reihenfolge: Oben-Links → Oben-Rechts → Unten-Rechts → Unten-Links\n"
                          "3. Drücken Sie 'C' zum Speichern oder ESC zum Abbrechen")
        self._update_status("Kalibrierung: Funktion über Taste 'C' verfügbar")
    
    def _reset_calibration(self):
        settings.transform_matrix = None
        settings.kalibrierungs_punkte = None
        state.changed = True
        self._update_status("Kalibrierung zurückgesetzt")
    
    def _set_camera(self, index: int):
        global camera
        camera.release()
        settings.camera_index = index
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_resolution[1])
            state.changed = True
            self._update_status(f"Kamera {index} ausgewählt")
        else:
            messagebox.showerror("Fehler", f"Kamera {index} konnte nicht geöffnet werden!")
            camera = cv2.VideoCapture(settings.camera_index)
    
    def _set_resolution(self, width: int, height: int):
        settings.camera_resolution = (width, height)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        state.changed = True
        self._update_status(f"Kamera-Auflösung: {width}x{height}")
    
    def _start_tts(self):
        if state.tts_thread_lock.locked():
            self._update_status("TTS läuft bereits...")
            return
        
        def tts_worker():
            state.tts_active = True
            self._update_status("OCR-Erkennung...")
            if state.last_processed_frame is not None:
                text = self.ocr_processor.recognize_text_fast(state.last_processed_frame)
                self._update_status("Sprachausgabe...")
                self.tts_manager.speak(text)
            state.tts_active = False
            state.tts_thread_lock.release()
        
        if state.tts_thread_lock.acquire(blocking=False):
            threading.Thread(target=tts_worker, daemon=True).start()
    
    def _stop_tts(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
        self._update_status("Wiedergabe gestoppt")
    
    def _set_tts_rate(self, rate: int):
        settings.tts_pyttsx_rate = rate
        if self.tts_manager._pyttsx_engine:
            self.tts_manager._pyttsx_engine.setProperty('rate', rate)
        state.changed = True
        self._update_status(f"TTS-Geschwindigkeit: {rate} WPM")
    
    def _set_tts_lang(self, lang: str):
        settings.tts_google_lang = lang
        state.changed = True
        self._update_status(f"TTS-Sprache: {lang}")
    
    def _show_shortcuts(self):
        shortcuts = """Tastenkürzel:

Numblock + / -  : Zoom vergrößern/verkleinern
Numblock *      : Schwarz-Weiß umschalten
Numblock 0      : Invertieren
Leertaste       : Text vorlesen
Pfeiltasten     : Helligkeit/Kontrast anpassen
A               : Auto-Helligkeit An/Aus
D               : Bild drehen (90°)
O               : OCR-Overlay An/Aus
M               : Schwarz-Weiß umschalten
I               : Invertieren
C               : Kalibrierung starten
Q               : Beenden"""
        messagebox.showinfo("Tastenkürzel", shortcuts)
    
    def _show_about(self):
        about = """Lesehilfe für Senioren

Version 1.6.1 (Windows)
Ersteller: Benjamin Deiss

Funktionen:
• Echtzeit-Kamera-Vorschau
• Zoom und Bildverbesserung
• OCR-Texterkennung
• Sprachausgabe (Online/Offline)
• Perspektiv-Kalibrierung"""
        messagebox.showinfo("Über Lesehilfe", about)
    
    def _update_status(self, text: str):
        self.status_var.set(text)
        state.status_text = text
        state.status_end_time = time.time() + 3
    
    # === Event-Handler ===
    
    def _on_key(self, event):
        key = event.keysym
        
        if key == 'space':
            self._start_tts()
        elif key == 'q' or key == 'Escape':
            self._on_close()
        elif key == 'plus' or key == 'KP_Add':
            self._set_zoom(min(settings.zoom_factor + 0.5, 8.0))
        elif key == 'minus' or key == 'KP_Subtract':
            self._set_zoom(max(settings.zoom_factor - 0.5, 1.0))
        elif key == 'asterisk' or key == 'KP_Multiply' or key == 'm':
            self.monochrome_var.set(not self.monochrome_var.get())
            self._toggle_monochrome()
        elif key == '0' or key == 'KP_0' or key == 'i':
            self.invert_var.set(not self.invert_var.get())
            self._toggle_invert()
        elif key == 'o':
            self.ocr_var.set(not self.ocr_var.get())
            self._toggle_ocr()
        elif key == 'a':
            self.auto_brightness_var.set(not self.auto_brightness_var.get())
            self._toggle_auto_brightness()
        elif key == 'd':
            self._set_rotation((settings.rotation + 90) % 360)
        elif key == 'Up':
            self._set_brightness(min(settings.brightness_beta + 5, 100))
        elif key == 'Down':
            self._set_brightness(max(settings.brightness_beta - 5, -100))
        elif key == 'Right':
            self._set_contrast(min(round(settings.contrast_alpha + 0.1, 1), 2.0))
        elif key == 'Left':
            self._set_contrast(max(round(settings.contrast_alpha - 0.1, 1), 0.5))
    
    def _on_resize(self, event):
        if event.widget == self.root:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w > 10 and h > 10:
                state.window_size = (w, h)
                global TARGET_ASPECT_RATIO
                TARGET_ASPECT_RATIO = w / h
    
    def _on_close(self):
        state.running = False
        if state.changed:
            SettingsManager.save(settings)
        camera.release()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        self.root.destroy()
    
    # === Haupt-Update-Schleife ===
    
    def _update_frame(self):
        if not state.running:
            return
        
        ret, frame = camera.read()
        if ret:
            # Verarbeite Frame
            processed = self.image_processor.process(frame)
            state.last_processed_frame = processed.copy()
            
            # OCR-Overlay
            if state.ocr_active:
                processed = self.ocr_processor.process_with_overlay(processed)
            
            # Invertierung
            if state.invert_contrast:
                processed = cv2.bitwise_not(processed)
            
            # Konvertiere für Tkinter
            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Skaliere auf Canvas-Größe
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Speichern wenn geändert
        if state.changed:
            SettingsManager.save(settings)
            state.changed = False
        
        # Nächstes Frame
        delay = 100 if state.tts_active else 30
        self.root.after(delay, self._update_frame)
    
    def run(self):
        self.root.mainloop()


# =============================================================================
# PROGRAMMSTART
# =============================================================================

def main():
    print("=" * 60)
    print("LESEHILFE FÜR SENIOREN - Windows Version 1.6.1")
    print("=" * 60)
    
    try:
        app = LesehilfeGUI()
        app.run()
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        print(f"\nFEHLER: {e}")
        print("\nStellen Sie sicher, dass:")
        print("  1. Eine USB-Kamera angeschlossen ist")
        print("  2. Tesseract-OCR installiert ist")
        print("  3. Die Python-Pakete installiert sind:")
        print("     pip install opencv-python numpy pytesseract pillow gtts requests pyttsx3 pygame")


if __name__ == "__main__":
    main()
