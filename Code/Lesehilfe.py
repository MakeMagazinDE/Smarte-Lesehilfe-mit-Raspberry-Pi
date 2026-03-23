#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
Projekt:   Lesehilfe für Senioren
Datei:     Lesehilfe.py

Ersteller: Benjamin Deiss
Version:   1.6.0
Datum:     23.01.2026

Beschreibung:
Echtzeit-Kamera-Vorschau mit Zoom, Kontrast- und
Helligkeitssteuerung, OCR-Texterkennung mit Tesseract,
Sprachausgabe mit Google TTS (online) und espeak-ng (offline),
GPIO- und Tastatursteuerung auf Raspberry Pi.
============================================================
"""

from __future__ import annotations

import cv2
from picamera2 import Picamera2
import numpy as np
import json
import os
import subprocess
from gpiozero import Button
import time
import pytesseract
import threading
import requests
from gtts import gTTS
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Callable, Any
from enum import IntEnum
from pathlib import Path
from contextlib import contextmanager
import logging

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
    """Bildqualitätsmodi für verschiedene Interpolations- und Schärfungsstufen"""
    LINEAR = 0
    LANCZOS = 1
    SHARPEN_LIGHT = 2
    SHARPEN_STRONG = 3


class KeyCode:
    """Tastencode-Definitionen für bessere Lesbarkeit"""
    ESC = 27
    ENTER = 13
    BACKSPACE = 8
    
    # Cursor-Tasten (OpenCV Codes)
    ARROW_UP = 82
    ARROW_DOWN = 84
    ARROW_LEFT = 81
    ARROW_RIGHT = 83
    
    # Numblock-Tasten (plattformabhängig, daher mehrere Optionen)
    NUM_DIVIDE = ord('/')  # Kontrast-Modus
    NUM_MULTIPLY = ord('*')  # Monochrom
    NUM_MINUS = ord('-')  # Zoom Out
    NUM_PLUS = ord('+')  # Zoom In
    NUM_0 = ord('0')  # Invert
    NUM_7 = ord('7')  # Exit
    NUM_8 = ord('8')  # Shutdown
    NUM_9 = ord('9')  # Reboot


# Dateipfade als Path-Objekte
TEMP_AUDIO_FILE = Path("/tmp/speak.mp3")
SETTINGS_FILE = Path("settings.json")
CALIBRATION_DATA_FILE = Path("calibration_data.json")

# Display-Konstanten (werden beim Start durch detect_screen_resolution() gesetzt)
DEFAULT_RESOLUTION = (1280, 1024)
OUTPUT_RESOLUTION: Tuple[int, int] = DEFAULT_RESOLUTION
TARGET_ASPECT_RATIO: float = 4 / 3


def detect_screen_resolution() -> Tuple[Tuple[int, int], float]:
    """
    Erkennt die aktuelle Bildschirmauflösung und berechnet das Seitenverhältnis.
    
    Verwendet mehrere Methoden in folgender Reihenfolge:
    1. xrandr (X11)
    2. fbset (Framebuffer)
    3. /sys/class/graphics/fb0 (Kernel)
    4. Fallback auf DEFAULT_RESOLUTION
    
    Returns:
        Tuple aus (Breite, Höhe) und Seitenverhältnis (4:3 oder 16:9)
    """
    import re
    
    resolution = None
    
    # Methode 1: xrandr (X11)
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            match = re.search(r'current\s+(\d+)\s*x\s*(\d+)', result.stdout)
            if match:
                resolution = (int(match.group(1)), int(match.group(2)))
                logger.info(f"Bildschirmauflösung erkannt (xrandr): {resolution[0]}x{resolution[1]}")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Methode 2: fbset (Framebuffer)
    if resolution is None:
        try:
            result = subprocess.run(
                ["fbset", "-s"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'geometry\s+(\d+)\s+(\d+)', result.stdout)
                if match:
                    resolution = (int(match.group(1)), int(match.group(2)))
                    logger.info(f"Bildschirmauflösung erkannt (fbset): {resolution[0]}x{resolution[1]}")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
    
    # Methode 3: Kernel Framebuffer
    if resolution is None:
        try:
            fb_path = Path("/sys/class/graphics/fb0/virtual_size")
            if fb_path.exists():
                content = fb_path.read_text().strip()
                parts = content.split(',')
                if len(parts) >= 2:
                    resolution = (int(parts[0]), int(parts[1]))
                    logger.info(f"Bildschirmauflösung erkannt (fb0): {resolution[0]}x{resolution[1]}")
        except Exception:
            pass
    
    # Fallback
    if resolution is None:
        resolution = DEFAULT_RESOLUTION
        logger.warning(f"Bildschirmauflösung nicht erkannt, verwende Fallback: {resolution[0]}x{resolution[1]}")
    
    # Seitenverhältnis berechnen (4:3 = 1.33, 16:9 = 1.78)
    ratio = resolution[0] / resolution[1]
    if ratio > 1.5:
        aspect_ratio = 16 / 9
        logger.info(f"Seitenverhältnis: 16:9 ({ratio:.2f})")
    else:
        aspect_ratio = 4 / 3
        logger.info(f"Seitenverhältnis: 4:3 ({ratio:.2f})")
    
    return resolution, aspect_ratio

# Zoom-Grenzen (min, max, step)
ZOOM_CONFIG = (1.0, 8.0, 0.5)

# Bildverarbeitungs-Konstanten
BRIGHTNESS_TARGET = 127
BRIGHTNESS_STEP = 12.7  # 10% Schritte
BRIGHTNESS_MAX = 75
CONTRAST_LOW_THRESHOLD = 40
CONTRAST_HIGH_VALUE = 1.4
HISTORY_SIZE = 5

# OCR-Konstanten
OCR_SIZE = (800, 600)
OCR_CONFIDENCE_THRESHOLD = 60
OCR_OVERLAY_ALPHA = 0.8


# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

@dataclass
class Settings:
    """Zentrale Einstellungsverwaltung mit Type Hints und Default Values"""
    zoom_factor: float = 1.0
    rotation: int = 0
    quality_mode: int = QualityMode.SHARPEN_STRONG
    contrast_alpha: float = 1.2
    brightness_beta: float = 0.0
    auto_brightness: bool = True
    color_order: str = "RGB"
    camera_resolution: Tuple[int, int] = (2304, 1296)
    monochrome_mode: bool = False
    
    # Kalibrierungsdaten (optional)
    kalibrierungs_punkte: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    
    # TTS-Einstellungen
    tts_espeak_lang: str = "de"
    tts_espeak_voice: str = "+f3"
    tts_espeak_speed: int = 150
    tts_espeak_volume: int = 100
    tts_google_lang: str = "de"
    tts_google_slow: bool = False
    
    def validate(self) -> None:
        """Validiert die Einstellungen"""
        if not (ZOOM_CONFIG[0] <= self.zoom_factor <= ZOOM_CONFIG[1]):
            raise ValueError(f"Zoom muss zwischen {ZOOM_CONFIG[0]} und {ZOOM_CONFIG[1]} liegen")
        if self.rotation not in [0, 90, 180, 270]:
            raise ValueError("Rotation muss 0, 90, 180 oder 270 Grad sein")


@dataclass
class ApplicationState:
    """Globaler Anwendungszustand"""
    running: bool = True
    changed: bool = False
    ocr_active: bool = False
    invert_contrast: bool = False
    
    # Status-Anzeige
    status_text: str = ""
    status_end_time: float = 0
    
    # OCR & TTS
    last_recognized_text: str = ""
    last_spoken_text: str = ""
    last_processed_frame: Optional[np.ndarray] = None
    tts_process: Optional[subprocess.Popen] = None
    tts_thread_lock: threading.Lock = field(default_factory=threading.Lock)
    tts_active: bool = False  # Reduzierte Frame-Rate während TTS
    
    # Bildverarbeitung
    brightness_history: list[float] = field(default_factory=list)
    contrast_history: list[float] = field(default_factory=list)
    
    # Kalibrierung
    clicked_points: list[Tuple[int, int]] = field(default_factory=list)


# Globale Instanzen (nach Initialisierung gesetzt)
settings: Settings
state: ApplicationState
picam2: Picamera2


# =============================================================================
# DATEI-MANAGEMENT
# =============================================================================

class SettingsManager:
    """Verwaltet Laden und Speichern von Einstellungen"""
    
    @staticmethod
    def load() -> Settings:
        """Lädt Einstellungen aus JSON-Datei"""
        if not SETTINGS_FILE.exists():
            logger.info("Keine Settings-Datei gefunden, verwende Standardwerte")
            return Settings()
        
        try:
            with SETTINGS_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Konvertiere Listen zu NumPy-Arrays
            if data.get("kalibrierungs_punkte") is not None:
                data["kalibrierungs_punkte"] = np.array(
                    data["kalibrierungs_punkte"], dtype=np.float32
                )
            if data.get("transform_matrix") is not None:
                data["transform_matrix"] = np.array(
                    data["transform_matrix"], dtype=np.float32
                )
            
            # Konvertiere camera_resolution zu Tuple
            if data.get("camera_resolution") and isinstance(data["camera_resolution"], list):
                data["camera_resolution"] = tuple(data["camera_resolution"])
            
            # Filtere nur bekannte Felder
            valid_fields = {k: v for k, v in data.items() 
                          if k in Settings.__annotations__}
            
            settings = Settings(**valid_fields)
            settings.validate()
            return settings
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Einstellungen: {e}")
            return Settings()
    
    @staticmethod
    def save(settings: Settings) -> None:
        """Speichert Einstellungen als JSON"""
        try:
            data = asdict(settings)
            
            # Konvertiere NumPy-Arrays zu Listen
            if data.get("kalibrierungs_punkte") is not None:
                data["kalibrierungs_punkte"] = data["kalibrierungs_punkte"].tolist()
            if data.get("transform_matrix") is not None:
                data["transform_matrix"] = data["transform_matrix"].tolist()
            
            # Konvertiere Tuple zu Liste
            if isinstance(data.get("camera_resolution"), tuple):
                data["camera_resolution"] = list(data["camera_resolution"])
            
            with SETTINGS_FILE.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            logger.debug("Einstellungen gespeichert")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")


class CalibrationManager:
    """Verwaltet Kamera-Kalibrierungsdaten"""
    
    @staticmethod
    def load() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Lädt Kamera-Kalibrierungsdaten für Fischaugen-Korrektur"""
        if not CALIBRATION_DATA_FILE.exists():
            logger.info("Keine Kalibrierungsdaten gefunden")
            return None, None
        
        try:
            with CALIBRATION_DATA_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
            dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
            
            logger.info("Kalibrierungsdaten geladen")
            return camera_matrix, dist_coeffs
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Kalibrierung: {e}")
            return None, None


# =============================================================================
# BILDVERARBEITUNG
# =============================================================================

class ImageProcessor:
    """Zentrale Bildverarbeitung mit Pipeline-Pattern"""
    
    def __init__(self, settings: Settings, state: ApplicationState):
        self.settings = settings
        self.state = state
    
    def process(
        self,
        frame: np.ndarray,
        apply_zoom: bool = True,
        apply_sharpening: bool = True,
        apply_color_correction: bool = True
    ) -> np.ndarray:
        """
        Zentrale Frame-Verarbeitung für maximale Schärfe
        
        Pipeline (hochauflösend bis nach Zoom):
        1. Rotation (in Original-Auflösung)
        2. Aspect-Ratio-Crop (in Original-Auflösung)
        3. Perspektiv-Korrektur/Warping
        4. Zoom-Crop (in hoher Auflösung!)
        5. Skalierung auf Ausgabeauflösung
        6. Monochrom-Umwandlung
        7. Helligkeit/Kontrast
        8. Schärfung
        9. Farbkorrektur
        """
        # 1-2. Rotation & Aspect-Crop
        frame = self._rotate(frame)
        frame = self._crop_to_aspect_ratio(frame)
        
        # 3. Perspektiv-Korrektur
        if self.settings.transform_matrix is not None:
            frame = self._apply_perspective_warp(frame)
        
        # 4. Zoom (in hoher Auflösung)
        if apply_zoom and self.settings.zoom_factor > 1.0:
            frame = self._apply_zoom(frame)
        
        # 5. Skalierung (NACH Zoom für maximale Schärfe)
        frame = self._scale_to_output(frame)
        
        # 6. Helligkeit/Kontrast (VOR Monochrom für bessere Schwellenwert-Erkennung)
        if apply_color_correction:
            frame = self._adjust_brightness_contrast(frame)
        
        # 7. Monochrom (NACH Helligkeits-/Kontrastanpassung)
        if self.settings.monochrome_mode:
            frame = self._convert_to_monochrome(frame)
        
        # 8. Schärfung
        if apply_sharpening and self.settings.quality_mode >= QualityMode.SHARPEN_LIGHT:
            frame = self._apply_sharpening(frame)
        
        # 9. Farbkorrektur (nur wenn nicht Monochrom)
        if apply_color_correction and self.settings.color_order != "RGB" and not self.settings.monochrome_mode:
            frame = self._convert_color_order(frame)
        
        return frame
    
    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        """Rotiert Frame um 0, 90, 180 oder 270 Grad"""
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        
        if self.settings.rotation in rotation_map:
            return cv2.rotate(frame, rotation_map[self.settings.rotation])
        return frame
    
    def _crop_to_aspect_ratio(self, frame: np.ndarray) -> np.ndarray:
        """Schneidet Frame auf Ziel-Seitenverhältnis zu"""
        h, w = frame.shape[:2]
        current_aspect = w / h
        
        if current_aspect > TARGET_ASPECT_RATIO:
            # Zu breit - schneide Seiten ab
            new_w = int(h * TARGET_ASPECT_RATIO)
            x1 = (w - new_w) // 2
            return frame[:, x1:x1+new_w]
        else:
            # Zu hoch - schneide oben/unten ab
            new_h = int(w / TARGET_ASPECT_RATIO)
            y1 = (h - new_h) // 2
            return frame[y1:y1+new_h, :]
    
    def _apply_perspective_warp(self, frame: np.ndarray) -> np.ndarray:
        """Wendet perspektivische Korrektur an"""
        try:
            h, w = frame.shape[:2]
            warp_h = h
            warp_w = int(warp_h * TARGET_ASPECT_RATIO)
            
            return cv2.warpPerspective(
                frame,
                self.settings.transform_matrix,
                (warp_w, warp_h)
            )
        except Exception as e:
            logger.error(f"Fehler bei Perspective Warp: {e}")
            return frame
    
    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Wendet Zoom durch zentrierten Crop an"""
        h, w = frame.shape[:2]
        zoom_w = int(w / self.settings.zoom_factor)
        zoom_h = int(h / self.settings.zoom_factor)
        x1 = (w - zoom_w) // 2
        y1 = (h - zoom_h) // 2
        return frame[y1:y1+zoom_h, x1:x1+zoom_w]
    
    def _scale_to_output(self, frame: np.ndarray) -> np.ndarray:
        """Skaliert auf finale Ausgabeauflösung"""
        interpolation = (cv2.INTER_LANCZOS4 
                        if self.settings.quality_mode > QualityMode.LINEAR 
                        else cv2.INTER_LINEAR)
        return cv2.resize(frame, OUTPUT_RESOLUTION, interpolation=interpolation)
    
    def _convert_to_monochrome(self, frame: np.ndarray) -> np.ndarray:
        """Konvertiert zu Schwarz-Weiß mit Otsu-Schwellenwert"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    def _adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Passt Helligkeit und Kontrast an (automatisch + manuell)"""
        if self.settings.auto_brightness:
            auto_alpha, auto_beta = self._calculate_auto_params(frame)
            final_alpha = self.settings.contrast_alpha * auto_alpha
            final_beta = auto_beta + self.settings.brightness_beta
        else:
            final_alpha = self.settings.contrast_alpha
            final_beta = self.settings.brightness_beta
        return cv2.convertScaleAbs(frame, alpha=final_alpha, beta=final_beta)
    
    def _calculate_auto_params(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Berechnet automatische Helligkeits- und Kontrastkorrektur
        Verwendet gleitenden Durchschnitt für sanfte Übergänge
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Helligkeit: Ziel = 127, in 10%-Stufen
        diff = BRIGHTNESS_TARGET - avg_brightness
        raw_beta = np.clip(
            round(diff / BRIGHTNESS_STEP) * BRIGHTNESS_STEP,
            -BRIGHTNESS_MAX, BRIGHTNESS_MAX
        )
        
        # Kontrast: Bei flachem Bild erhöhen
        raw_alpha = (CONTRAST_HIGH_VALUE 
                    if std_dev < CONTRAST_LOW_THRESHOLD 
                    else 1.2)
        
        # Glättung durch Moving Average
        self.state.brightness_history.append(raw_beta)
        self.state.contrast_history.append(raw_alpha)
        
        if len(self.state.brightness_history) > HISTORY_SIZE:
            self.state.brightness_history.pop(0)
            self.state.contrast_history.pop(0)
        
        avg_alpha = sum(self.state.contrast_history) / len(self.state.contrast_history)
        avg_beta = sum(self.state.brightness_history) / len(self.state.brightness_history)
        
        return avg_alpha, avg_beta
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Wendet Unscharf-Maskierung an"""
        sigma = (1.0 if self.settings.quality_mode == QualityMode.SHARPEN_LIGHT 
                else 2.0)
        weight = (1.3 if self.settings.quality_mode == QualityMode.SHARPEN_LIGHT 
                 else 1.8)
        
        blur = cv2.GaussianBlur(frame, (0, 0), sigma)
        return cv2.addWeighted(frame, weight, blur, -(weight - 1.0), 0)
    
    def _convert_color_order(self, frame: np.ndarray) -> np.ndarray:
        """Konvertiert Farbordnung (z.B. BGR zu RGB)"""
        conversion_name = f"COLOR_{self.settings.color_order}2RGB"
        conversion = getattr(cv2, conversion_name, None)
        
        if conversion:
            return cv2.cvtColor(frame, conversion)
        return frame


# =============================================================================
# OCR & SPRACHAUSGABE
# =============================================================================

class OCRProcessor:
    """Verarbeitet Texterkennung mit Tesseract"""
    
    def __init__(self, state: ApplicationState):
        self.state = state
    
    def process_with_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Führt OCR durch und überlagert erkannten Text"""
        # Für OCR verkleinern (Performance)
        ocr_frame = cv2.resize(frame, OCR_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ocr_frame, cv2.COLOR_RGB2GRAY)
        
        try:
            data = pytesseract.image_to_data(
                gray,
                lang='deu',
                output_type=pytesseract.Output.DICT,
                config='--psm 3'
            )
        except Exception as e:
            logger.error(f"OCR-Fehler: {e}")
            self.state.last_recognized_text = ""
            return frame
        
        # Erstelle Overlay
        overlay = frame.copy()
        recognized_words = []
        
        scale_x = frame.shape[1] / OCR_SIZE[0]
        scale_y = frame.shape[0] / OCR_SIZE[1]
        
        # Dynamische Farben für Invertierung
        bg_color = (0, 0, 0) if self.state.invert_contrast else (255, 255, 255)
        text_color = (255, 255, 255) if self.state.invert_contrast else (0, 0, 0)
        
        # Font-Parameter
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, conf in enumerate(data['conf']):
            if int(float(conf)) > OCR_CONFIDENCE_THRESHOLD and data['text'][i].strip():
                # Text ohne Encoding-Probleme
                text = data['text'][i].strip()
                
                # Filtere nur komplett unleserliche Wörter (NUR Fragezeichen/Sonderzeichen)
                # Legitime Fragezeichen in Sätzen bleiben erhalten
                cleaned = text.replace('?', '').replace('¿', '').replace('�', '')
                if len(cleaned) == 0:
                    # Wort besteht nur aus unlesbaren Zeichen - überspringen
                    continue
                
                recognized_words.append(text)
                
                # Skaliere Koordinaten zurück
                x = int(data['left'][i] * scale_x)
                y = int(data['top'][i] * scale_y)
                w = int(data['width'][i] * scale_x)
                h = int(data['height'][i] * scale_y)
                
                # Berechne optimale Font-Größe
                # Basiert auf Höhe der Box, aber mit Minimum für Lesbarkeit
                font_scale = max(h / 35.0, 0.8)  # Mindestens 0.8, sonst proportional
                
                # Berechne tatsächliche Text-Größe für besseres Padding
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness=1
                )
                
                # Padding für bessere Abdeckung (10% der Höhe)
                padding_x = int(h * 0.15)
                padding_y = int(h * 0.15)
                
                # Zeichne Hintergrund-Rechteck mit Padding
                box_x1 = max(0, x - padding_x)
                box_y1 = max(0, y - padding_y)
                box_x2 = min(frame.shape[1], x + w + padding_x)
                box_y2 = min(frame.shape[0], y + h + padding_y)
                
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
                
                # Berechne optimale Dicke (ähnlich wie OSD: proportional zur Schriftgröße)
                thickness = max(int(font_scale * 2.5), 3)  # Mindestens 3, sonst proportional
                
                # Zentriere Text vertikal in der Box
                text_y = y + h - padding_y + int((padding_y * 2 - text_h) / 2)
                text_x = x + padding_x // 2
                
                # Zeichne Text mit dicker Linie (wie OSD)
                cv2.putText(
                    overlay, text, (text_x, text_y),
                    font, font_scale,
                    text_color, thickness, cv2.LINE_AA
                )
        
        self.state.last_recognized_text = " ".join(recognized_words)
        return cv2.addWeighted(overlay, OCR_OVERLAY_ALPHA, frame, 1 - OCR_OVERLAY_ALPHA, 0)


class TTSManager:
    """Verwaltet Text-to-Speech (online und offline)"""
    
    def __init__(self, settings: Settings, state: ApplicationState):
        self.settings = settings
        self.state = state
    
    def speak(self, text: str) -> None:
        """Spricht Text aus mit Online/Offline-Fallback"""
        # Stoppe laufende Ausgabe
        self._stop_current_playback()
        
        text_to_speak = text.strip() or "Kein Text erkannt."
        
        # Cache-Check
        if self._use_cached_audio(text_to_speak):
            return
        
        # Online: Google TTS
        if self._check_internet():
            if self._speak_with_gtts(text_to_speak):
                return
        
        # Offline: espeak-ng
        self._speak_with_espeak(text_to_speak)
    
    def _stop_current_playback(self) -> None:
        """Stoppt aktuell laufende Sprachausgabe"""
        if self.state.tts_process is not None:
            self.state.tts_process.terminate()
            self.state.tts_process.wait()
            self.state.tts_process = None
    
    def _use_cached_audio(self, text: str) -> bool:
        """Prüft und nutzt Cache wenn möglich"""
        if text == self.state.last_spoken_text and TEMP_AUDIO_FILE.exists():
            UIHelper.update_status("Audio (gecached)", self.state)
            try:
                self.state.tts_process = subprocess.Popen(
                    ["mpg123", "-q", str(TEMP_AUDIO_FILE)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            except FileNotFoundError:
                logger.error("mpg123 nicht gefunden")
        return False
    
    def _check_internet(self, timeout: int = 3) -> bool:
        """Prüft Internetverbindung"""
        try:
            requests.get("http://www.google.com", timeout=timeout)
            return True
        except:
            return False
    
    def _speak_with_gtts(self, text: str) -> bool:
        """Sprachausgabe mit Google TTS"""
        try:
            UIHelper.update_status("Google TTS laeuft...", self.state, duration=10)
            
            tts = gTTS(
                text=text,
                lang=self.settings.tts_google_lang,
                slow=self.settings.tts_google_slow
            )
            tts.save(str(TEMP_AUDIO_FILE))
            
            self.state.tts_process = subprocess.Popen(
                ["mpg123", "-q", str(TEMP_AUDIO_FILE)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.state.last_spoken_text = text
            return True
            
        except Exception as e:
            logger.error(f"Google TTS Fehler: {e}")
            return False
    
    def _speak_with_espeak(self, text: str) -> None:
        """Sprachausgabe mit espeak-ng"""
        try:
            UIHelper.update_status("Offline TTS (espeak)", self.state, duration=10)
            
            cmd = [
                "espeak-ng",
                f"-v{self.settings.tts_espeak_lang}{self.settings.tts_espeak_voice}",
                f"-s{self.settings.tts_espeak_speed}",
                f"-a{self.settings.tts_espeak_volume}",
                text
            ]
            self.state.tts_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.state.last_spoken_text = ""
            
        except Exception as e:
            logger.error(f"espeak-ng Fehler: {e}")


# =============================================================================
# UI & HILFS-FUNKTIONEN
# =============================================================================

class UIHelper:
    """Hilfs-Funktionen für Benutzeroberfläche"""
    
    @staticmethod
    def update_status(text: str, state: ApplicationState, duration: int = 3) -> None:
        """Aktualisiert Status-Text mit Timeout"""
        state.status_text = text
        state.status_end_time = time.time() + duration
        logger.debug(f"Status: {text}")
    
    @staticmethod
    def draw_status_overlay(frame: np.ndarray, state: ApplicationState) -> np.ndarray:
        """Zeichnet Status-Text"""
        if not state.status_text or time.time() >= state.status_end_time:
            return frame
        
        # Schriftgröße und Dicke (3x größer als Original)
        font_scale, font_thickness = 2.1, 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Dynamische Farben für Invertierung
        bg_color = (0, 0, 0) if state.invert_contrast else (255, 255, 255)
        text_color = (255, 255, 255) if state.invert_contrast else (0, 0, 0)
        
        # Berechne Textgröße
        (text_w, text_h), _ = cv2.getTextSize(
            state.status_text, font, font_scale, font_thickness
        )
        
        # Zeichne Hintergrund-Kasten
        padding = 20
        rect_y1 = OUTPUT_RESOLUTION[1] - text_h - padding * 2
        rect_y2 = OUTPUT_RESOLUTION[1] - padding // 2
        
        cv2.rectangle(
            frame,
            (padding // 2, rect_y1),
            (text_w + padding * 2, rect_y2),
            bg_color, -1
        )
        
        # Zeichne Text
        cv2.putText(
            frame, state.status_text,
            (padding, rect_y2 - padding),
            font, font_scale, text_color,
            font_thickness, cv2.LINE_AA
        )
        
        return frame


class SystemHelper:
    """System-Operationen (Shutdown, Reboot, etc.)"""
    
    @staticmethod
    def shutdown() -> None:
        """Fährt System herunter"""
        logger.info("System wird heruntergefahren...")
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])
    
    @staticmethod
    def reboot() -> None:
        """Startet System neu"""
        logger.info("System wird neu gestartet...")
        subprocess.Popen(["sudo", "reboot"])
    
    @staticmethod
    @contextmanager
    def hide_cursor():
        """Context Manager zum Verstecken des Mauszeigers"""
        try:
            # Starte unclutter
            subprocess.Popen(
                ["unclutter", "-idle", "0.1", "-root"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            yield
        finally:
            # Stoppe unclutter
            subprocess.run(
                ["pkill", "unclutter"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )


# =============================================================================
# KALIBRIERUNG
# =============================================================================

class CalibrationHelper:
    """Interaktive perspektivische Kalibrierung"""
    
    def __init__(
        self,
        settings: Settings,
        state: ApplicationState,
        image_processor: ImageProcessor
    ):
        self.settings = settings
        self.state = state
        self.image_processor = image_processor
    
    def run(self) -> None:
        """Startet interaktive Kalibrierung"""
        logger.info("Starte Kalibrierung")
        print("=" * 60)
        print("PERSPEKTIVISCHE KALIBRIERUNG")
        print("=" * 60)
        print("Klicke 4 Eckpunkte eines Rechtecks (z.B. DIN A4 Blatt)")
        print("Reihenfolge: Oben-Links → Oben-Rechts → Unten-Rechts → Unten-Links")
        print("C = Bestätigen | ESC = Abbrechen")
        print("=" * 60)
        
        UIHelper.update_status("Kalibrierung gestartet", self.state, duration=5)
        
        # Zeige Mauszeiger
        subprocess.run(
            ["pkill", "unclutter"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        self.state.clicked_points = []
        window_name = "Kalibrierung"
        
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        try:
            self._calibration_loop(window_name)
        finally:
            cv2.destroyWindow(window_name)
            # Verstecke Mauszeiger wieder
            subprocess.Popen(
                ["unclutter", "-idle", "0.1", "-root"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Callback für Mausklicks"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.state.clicked_points) < 4:
            self.state.clicked_points.append((x, y))
            logger.info(f"Punkt {len(self.state.clicked_points)}/4: ({x}, {y})")
    
    def _calibration_loop(self, window_name: str) -> None:
        """Hauptschleife der Kalibrierung"""
        while True:
            frame = picam2.capture_array()
            
            # NUR Rotation, Aspect-Crop und Skalierung - KEIN Warping, kein Zoom
            # Damit die Punkte auf dem unverarbeiteten Bild gesetzt werden
            
            # 1. Rotation
            if self.settings.rotation != 0:
                rotation_map = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }
                if self.settings.rotation in rotation_map:
                    frame = cv2.rotate(frame, rotation_map[self.settings.rotation])
            
            # 2. Aspect-Crop
            h, w = frame.shape[:2]
            current_aspect = w / h
            if current_aspect > TARGET_ASPECT_RATIO:
                new_w = int(h * TARGET_ASPECT_RATIO)
                x1 = (w - new_w) // 2
                frame = frame[:, x1:x1+new_w]
            else:
                new_h = int(w / TARGET_ASPECT_RATIO)
                y1 = (h - new_h) // 2
                frame = frame[y1:y1+new_h, :]
            
            # 3. Nur auf Bildschirmgröße skalieren
            frame_display = cv2.resize(frame, OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            
            # Zeichne geklickte Punkte
            self._draw_points(frame_display)
            self._draw_instructions(frame_display)
            
            cv2.imshow(window_name, frame_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == KeyCode.ESC:
                logger.info("Kalibrierung abgebrochen")
                UIHelper.update_status("Abgebrochen", self.state)
                self.state.clicked_points = []
                break
            elif (key == ord('c') or len(self.state.clicked_points) == 4) and len(self.state.clicked_points) == 4:
                if self._calculate_transform():
                    break
    
    def _draw_points(self, frame: np.ndarray) -> None:
        """Zeichnet geklickte Punkte auf Frame"""
        for i, (x, y) in enumerate(self.state.clicked_points):
            cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
            cv2.putText(
                frame, str(i + 1), (x + 25, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4
            )
    
    def _draw_instructions(self, frame: np.ndarray) -> None:
        """Zeichnet Anweisungen auf Frame"""
        cv2.putText(
            frame, "Ecken des Blattes anklicken",
            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )
        cv2.putText(
            frame, "C=Speichern | ESC=Abbruch",
            (50, OUTPUT_RESOLUTION[1] - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4
        )
        cv2.putText(
            frame, f"Punkte: {len(self.state.clicked_points)}/4",
            (50, OUTPUT_RESOLUTION[1] - 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4
        )
    
    def _calculate_transform(self) -> bool:
        """Berechnet Transformationsmatrix aus Punkten"""
        try:
            # Hole Original-Frame für Skalierungsberechnung
            frame_temp = picam2.capture_array()
            
            # Simuliere Rotation und Crop
            frame_temp = self.image_processor._rotate(frame_temp)
            h_orig, w_orig = frame_temp.shape[:2]
            current_aspect = w_orig / h_orig
            
            # Berechne Crop-Dimensionen
            if current_aspect > TARGET_ASPECT_RATIO:
                w_cropped = int(h_orig * TARGET_ASPECT_RATIO)
                h_cropped = h_orig
            else:
                w_cropped = w_orig
                h_cropped = int(w_orig / TARGET_ASPECT_RATIO)
            
            # Skaliere Punkte zurück auf Original
            scale_x = w_cropped / OUTPUT_RESOLUTION[0]
            scale_y = h_cropped / OUTPUT_RESOLUTION[1]
            
            src_pts_orig = np.array([
                [x * scale_x, y * scale_y]
                for x, y in self.state.clicked_points
            ], dtype=np.float32)
            
            # Ziel-Punkte
            dst_pts = np.float32([
                [0, 0],
                [w_cropped, 0],
                [w_cropped, h_cropped],
                [0, h_cropped]
            ])
            
            # Berechne Transformation
            self.settings.transform_matrix = cv2.getPerspectiveTransform(
                src_pts_orig, dst_pts
            )
            self.settings.kalibrierungs_punkte = np.array(
                self.state.clicked_points, dtype=np.float32
            )
            self.state.changed = True
            
            logger.info(f"Kalibrierung erfolgreich! Crop: {w_cropped}x{h_cropped}")
            UIHelper.update_status("Gespeichert!", self.state, duration=5)
            return True
            
        except Exception as e:
            logger.error(f"Kalibrierungsfehler: {e}")
            UIHelper.update_status("Fehler!", self.state, duration=5)
            self.state.clicked_points = []
            return False


# =============================================================================
# EVENT-HANDLER
# =============================================================================

class EventHandler:
    """Zentrale Event-Behandlung für Tastatur und GPIO"""
    
    def __init__(
        self,
        settings: Settings,
        state: ApplicationState,
        ocr_processor: OCRProcessor,
        tts_manager: TTSManager,
        calibration_helper: CalibrationHelper
    ):
        self.settings = settings
        self.state = state
        self.ocr_processor = ocr_processor
        self.tts_manager = tts_manager
        self.calibration_helper = calibration_helper
    
    def handle_keyboard(self, key: int) -> None:
        """Verarbeitet Tastatureingaben"""
        # Key-Mapping mit neuen Numblock-Tasten
        key_map: Dict[int, Callable[[], None]] = {
            # Numblock (neue Belegung)
            KeyCode.NUM_PLUS: self.zoom_in,  # + : Zoom In
            KeyCode.NUM_MINUS: self.zoom_out,  # - : Zoom Out
            KeyCode.NUM_DIVIDE: self.cycle_quality,  # / : Kontrast-Modus
            KeyCode.NUM_MULTIPLY: self.toggle_monochrome,  # * : Monochrom
            KeyCode.NUM_0: self.toggle_invert,  # 0 : Invert
            KeyCode.NUM_7: self.exit,  # 7 : Exit
            KeyCode.NUM_8: self.shutdown,  # 8 : Shutdown
            KeyCode.NUM_9: self.reboot,  # 9 : Reboot
            KeyCode.ENTER: self.start_ocr_tts_thread,  # Enter : Vorlesen
            
            # Cursor-Tasten für Helligkeit und Kontrast
            KeyCode.ARROW_UP: self.brightness_up,      # Pfeil hoch : Helligkeit +
            KeyCode.ARROW_DOWN: self.brightness_down,  # Pfeil runter : Helligkeit -
            KeyCode.ARROW_RIGHT: self.contrast_up,     # Pfeil rechts : Kontrast +
            KeyCode.ARROW_LEFT: self.contrast_down,    # Pfeil links : Kontrast -
            
            # Buchstaben-Tasten
            ord('a'): self.toggle_auto_brightness,
            ord('d'): self.rotate,
            ord('c'): lambda: self.calibration_helper.run(),
            ord('o'): self.toggle_ocr,
            ord('l'): self.cycle_quality,  # l : Qualitätsmodus wechseln
            ord(' '): self.start_ocr_tts_thread,
            ord('q'): self.exit,
            ord('s'): self.shutdown,
            ord('r'): self.reboot,
            ord('i'): self.toggle_invert,
            ord('m'): self.toggle_monochrome,
            ord('.'): self.toggle_invert,
            ord(','): self.toggle_invert,
            KeyCode.BACKSPACE: self.toggle_ocr,
        }
        
        handler = key_map.get(key)
        if handler:
            handler()
    
    # Event-Handler-Methoden
    def zoom_in(self) -> None:
        """Vergrößert Ansicht"""
        min_zoom, max_zoom, step = ZOOM_CONFIG
        if self.settings.zoom_factor < max_zoom:
            self.settings.zoom_factor = min(self.settings.zoom_factor + step, max_zoom)
            UIHelper.update_status(f"Zoom: {self.settings.zoom_factor:.1f}x", self.state)
            self.state.changed = True
    
    def zoom_out(self) -> None:
        """Verkleinert Ansicht"""
        min_zoom, max_zoom, step = ZOOM_CONFIG
        if self.settings.zoom_factor > min_zoom:
            self.settings.zoom_factor = max(self.settings.zoom_factor - step, min_zoom)
            UIHelper.update_status(f"Zoom: {self.settings.zoom_factor:.1f}x", self.state)
            self.state.changed = True
    
    def rotate(self) -> None:
        """Rotiert Bild um 90°"""
        self.settings.rotation = (self.settings.rotation + 90) % 360
        UIHelper.update_status(f"Rotation: {self.settings.rotation} Grad", self.state)
        self.state.changed = True
    
    def cycle_quality(self) -> None:
        """Wechselt Qualitätsmodus"""
        self.settings.quality_mode = (self.settings.quality_mode + 1) % 4
        quality_names = ["Linear", "Lanczos", "Schaerfe 1", "Schaerfe 2"]
        UIHelper.update_status(f"Modus: {quality_names[self.settings.quality_mode]}", self.state)
        self.state.changed = True
    
    def toggle_invert(self) -> None:
        """Invertiert Kontrast"""
        self.state.invert_contrast = not self.state.invert_contrast
        status = "Invertiert" if self.state.invert_contrast else "Normal"
        UIHelper.update_status(status, self.state)
    
    def brightness_up(self) -> None:
        """Erhöht Helligkeit um 5 (max 100)"""
        if self.settings.brightness_beta < 100:
            self.settings.brightness_beta = min(self.settings.brightness_beta + 5, 100)
            UIHelper.update_status(f"Helligkeit: {self.settings.brightness_beta:+.0f}", self.state)
            self.state.changed = True
    
    def brightness_down(self) -> None:
        """Verringert Helligkeit um 5 (min -100)"""
        if self.settings.brightness_beta > -100:
            self.settings.brightness_beta = max(self.settings.brightness_beta - 5, -100)
            UIHelper.update_status(f"Helligkeit: {self.settings.brightness_beta:+.0f}", self.state)
            self.state.changed = True
    
    def contrast_up(self) -> None:
        """Erhöht Kontrast um 0.1 (max 2.0)"""
        if self.settings.contrast_alpha < 2.0:
            self.settings.contrast_alpha = min(self.settings.contrast_alpha + 0.1, 2.0)
            self.settings.contrast_alpha = round(self.settings.contrast_alpha, 1)
            UIHelper.update_status(f"Kontrast: {self.settings.contrast_alpha:.1f}", self.state)
            self.state.changed = True
    
    def contrast_down(self) -> None:
        """Verringert Kontrast um 0.1 (min 0.5)"""
        if self.settings.contrast_alpha > 0.5:
            self.settings.contrast_alpha = max(self.settings.contrast_alpha - 0.1, 0.5)
            self.settings.contrast_alpha = round(self.settings.contrast_alpha, 1)
            UIHelper.update_status(f"Kontrast: {self.settings.contrast_alpha:.1f}", self.state)
            self.state.changed = True
    
    def toggle_auto_brightness(self) -> None:
        """Schaltet automatische Helligkeitsanpassung um"""
        self.settings.auto_brightness = not self.settings.auto_brightness
        status = "Auto-Helligkeit An" if self.settings.auto_brightness else "Auto-Helligkeit Aus"
        UIHelper.update_status(status, self.state)
        self.state.changed = True
    
    def toggle_monochrome(self) -> None:
        """Schaltet Schwarz-Weiß-Modus um"""
        self.settings.monochrome_mode = not self.settings.monochrome_mode
        status = "Schwarz-Weiss An" if self.settings.monochrome_mode else "Schwarz-Weiss Aus"
        UIHelper.update_status(status, self.state)
        self.state.changed = True
    
    def toggle_ocr(self) -> None:
        """Schaltet OCR-Overlay um"""
        self.state.ocr_active = not self.state.ocr_active
        status = "OCR An" if self.state.ocr_active else "OCR Aus"
        UIHelper.update_status(status, self.state)
    
    def start_ocr_tts_thread(self) -> None:
        """Startet OCR+TTS in separatem Thread"""
        # Stoppe laufende Ausgabe
        if self.state.tts_process is not None and self.state.tts_process.poll() is None:
            UIHelper.update_status("Sprachausgabe gestoppt", self.state)
            self.state.tts_process.terminate()
            self.state.tts_process.wait()
            self.state.tts_process = None
            if self.state.tts_thread_lock.locked():
                self.state.tts_thread_lock.release()
            return
        
        # Starte neuen Thread
        if self.state.tts_thread_lock.acquire(blocking=False):
            thread = threading.Thread(target=self._ocr_tts_worker, daemon=True)
            thread.start()
        else:
            UIHelper.update_status("OCR/TTS laeuft bereits...", self.state)
    
    def _ocr_tts_worker(self) -> None:
        """Worker-Thread für OCR + Sprachausgabe"""
        try:
            self.state.tts_active = True
            UIHelper.update_status("OCR-Erkennung laeuft...", self.state, duration=10)
            
            # Verwende gecachtes Frame falls vorhanden
            if self.state.last_processed_frame is not None:
                processed = self.state.last_processed_frame.copy()
            else:
                frame = picam2.capture_array()
                processed = ImageProcessor(self.settings, self.state).process(
                    frame, apply_sharpening=False
                )
            
            self.ocr_processor.process_with_overlay(processed)
            UIHelper.update_status("Sprachausgabe...", self.state, duration=30)
            
            self.tts_manager.speak(self.state.last_recognized_text)
            
        except Exception as e:
            logger.error(f"Fehler in OCR-Thread: {e}")
            UIHelper.update_status("OCR Fehler", self.state)
        finally:
            self.state.tts_active = False
            if self.state.tts_thread_lock.locked():
                self.state.tts_thread_lock.release()
    
    def exit(self) -> None:
        """Beendet Programm"""
        logger.info("Programm wird beendet")
        UIHelper.update_status("Beende Programm...", self.state)
        self.state.running = False
    
    def shutdown(self) -> None:
        """Fährt System herunter"""
        UIHelper.update_status("Fahre herunter...", self.state)
        self.state.running = False
        SystemHelper.shutdown()
    
    def reboot(self) -> None:
        """Startet System neu"""
        UIHelper.update_status("Starte neu...", self.state)
        self.state.running = False
        SystemHelper.reboot()


# =============================================================================
# GPIO-SETUP
# =============================================================================

class GPIOManager:
    """Verwaltet GPIO-Button-Konfiguration"""
    
    def __init__(self, event_handler: EventHandler):
        self.event_handler = event_handler
        self.buttons: list[Button] = []
    
    def setup(self) -> None:
        """Konfiguriert alle GPIO-Buttons"""
        try:
            button_config = {
                5: self.event_handler.zoom_in,
                6: self.event_handler.zoom_out,
                13: self.event_handler.rotate,
                19: self.event_handler.cycle_quality,
                23: self.event_handler.toggle_invert,
                22: self.event_handler.toggle_monochrome,
                24: lambda: self.event_handler.calibration_helper.run(),
                18: self.event_handler.toggle_ocr,
                17: self.event_handler.start_ocr_tts_thread,
                26: self.event_handler.exit,
                20: self.event_handler.shutdown,
                21: self.event_handler.reboot,
            }
            
            for pin, handler in button_config.items():
                btn = Button(pin, pull_up=True)
                btn.when_pressed = handler
                self.buttons.append(btn)
            
            logger.info("GPIO-Buttons konfiguriert")
            
        except Exception as e:
            logger.warning(f"GPIO nicht verfügbar: {e}")


# =============================================================================
# HAUPT-ANWENDUNG
# =============================================================================

class Application:
    """Haupt-Anwendungsklasse"""
    
    def __init__(self):
        self.settings: Settings
        self.state: ApplicationState
        self.image_processor: ImageProcessor
        self.ocr_processor: OCRProcessor
        self.tts_manager: TTSManager
        self.event_handler: EventHandler
        self.gpio_manager: GPIOManager
        self.calibration_helper: CalibrationHelper
    
    def initialize(self) -> None:
        """Initialisiert alle Komponenten"""
        global settings, state, picam2, OUTPUT_RESOLUTION, TARGET_ASPECT_RATIO
        
        print("=" * 60)
        print("LESEHILFE FÜR SENIOREN - Version 1.6.0")
        print("=" * 60)
        
        # Erkenne Bildschirmauflösung und Seitenverhältnis
        OUTPUT_RESOLUTION, TARGET_ASPECT_RATIO = detect_screen_resolution()
        print(f"Bildschirm: {OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]}")
        print(f"Seitenverhältnis: {'16:9' if TARGET_ASPECT_RATIO > 1.5 else '4:3'}")
        
        # Lade Einstellungen
        self.settings = settings = SettingsManager.load()
        self.state = state = ApplicationState()
        
        # Initialisiere Kamera
        logger.info("Initialisiere Kamera...")
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": self.settings.camera_resolution, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        # Initialisiere Komponenten
        self.image_processor = ImageProcessor(self.settings, self.state)
        self.ocr_processor = OCRProcessor(self.state)
        self.tts_manager = TTSManager(self.settings, self.state)
        self.calibration_helper = CalibrationHelper(
            self.settings, self.state, self.image_processor
        )
        self.event_handler = EventHandler(
            self.settings, self.state,
            self.ocr_processor, self.tts_manager,
            self.calibration_helper
        )
        self.gpio_manager = GPIOManager(self.event_handler)
        
        # Setup GPIO
        self.gpio_manager.setup()
        
        # Erstelle Vollbild-Fenster
        cv2.namedWindow("Vorschau", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Vorschau", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        logger.info("System bereit!")
        print("=" * 60)
        print("Tastenbelegung:")
        print("  Numblock + / -  : Zoom")
        print("  Numblock /      : Qualitätsmodus")
        print("  Numblock *      : Monochrom")
        print("  Numblock 0      : Invertieren")
        print("  Numblock Enter  : Text vorlesen")
        print("  Numblock 7      : Beenden")
        print("  Numblock 8      : Herunterfahren")
        print("  Numblock 9      : Neustart")
        print("=" * 60)
    
    def run(self) -> None:
        """Hauptschleife der Anwendung"""
        logger.info("Starte Hauptschleife")
        
        with SystemHelper.hide_cursor():
            while self.state.running:
                self._process_frame()
                self._handle_input()
                self._save_if_changed()
    
    def _process_frame(self) -> None:
        """Verarbeitet und zeigt ein Frame"""
        # Reduzierte Frame-Rate während TTS für bessere Performance
        if self.state.tts_active:
            time.sleep(0.25)
        
        # Erfasse Frame
        frame = picam2.capture_array()
        
        # Verarbeite Frame
        frame = self.image_processor.process(frame)
        
        # Cache für OCR-TTS
        self.state.last_processed_frame = frame.copy()
        
        # OCR-Overlay
        if self.state.ocr_active:
            frame = self.ocr_processor.process_with_overlay(frame)
        
        # Invertierung
        if self.state.invert_contrast:
            frame = cv2.bitwise_not(frame)
        
        # Status-Overlay
        frame = UIHelper.draw_status_overlay(frame, self.state)
        
        # Zeige Frame
        cv2.imshow("Vorschau", frame)
    
    def _handle_input(self) -> None:
        """Verarbeitet Tastatureingaben"""
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self.event_handler.handle_keyboard(key)
    
    def _save_if_changed(self) -> None:
        """Speichert Einstellungen bei Änderungen"""
        if self.state.changed:
            SettingsManager.save(self.settings)
            self.state.changed = False
    
    def cleanup(self) -> None:
        """Räumt Ressourcen auf"""
        logger.info("Räume auf...")
        
        # Stoppe TTS
        if self.state.tts_process is not None:
            self.state.tts_process.terminate()
            self.state.tts_process.wait()
        
        # Schließe Fenster und Kamera
        cv2.destroyAllWindows()
        picam2.stop()
        
        logger.info("Programm beendet")


# =============================================================================
# PROGRAMMSTART
# =============================================================================

def main() -> None:
    """Hauptfunktion"""
    app = Application()
    
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        logger.info("Programm durch Benutzer unterbrochen (Ctrl+C)")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()