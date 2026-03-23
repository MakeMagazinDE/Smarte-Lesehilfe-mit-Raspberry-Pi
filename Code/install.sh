#!/bin/bash

# --- VARIANTE MIT SUDO PRÜFUNG ---
if [ "$EUID" -ne 0 ]
  then echo "Bitte als Root ausführen (sudo ./install.sh)"
  exit
fi

# Der Benutzer 'pi' wird als Standard für den Autostart angenommen.
# Passen Sie dies an, falls Sie einen anderen Benutzer verwenden.
USER_NAME="pi"
USER_HOME="/home/$USER_NAME"
PROJECT_DIR="$USER_HOME/Lesehilfe"

echo "Starte Installation..."

# 1. Systempakete installieren
echo "Aktualisiere System und installiere notwendige Pakete..."
apt update
apt install -y \
  tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng \
  espeak-ng mpg123 unclutter \
  libblas-dev liblapack-dev \
  libcap-dev \
  python3-venv python3-pip \
  python3-libcamera \
  fonts-dejavu fonts-dejavu-core fonts-dejavu-extra

# 2. Virtuelle Umgebung (Venv) erstellen und aktivieren
echo "Erstelle und konfiguriere virtuelle Umgebung (venv)..."
# Alte Venv löschen, um Konsistenz zu gewährleisten
rm -rf "$PROJECT_DIR/venv"

# Venv neu erstellen mit Zugriff auf Systempakete (für python3-libcamera)
# ACHTUNG: Der Venv-Befehl muss als normaler Benutzer ausgeführt werden, da die Venv im Home-Verzeichnis des Benutzers liegt.
sudo -u "$USER_NAME" python3 -m venv --system-site-packages "$PROJECT_DIR/venv"

# Pakete in der Venv installieren
# Zuerst aktivieren (source funktioniert nicht mit sudo), daher direkter Aufruf des Python-Interpreters der Venv.
# Die `requirements.txt` muss im Projektordner existieren.
sudo -u "$USER_NAME" "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# Pillow für Unicode-Textdarstellung (Umlaute ä, ö, ü, ß)
echo "Installiere Pillow für Unicode-Unterstützung..."
sudo -u "$USER_NAME" "$PROJECT_DIR/venv/bin/pip" install Pillow

# AUTOSTART KONFIGURATION
echo "Erstelle Autostart-Dateien für den Benutzer '$USER_NAME'..."

# 3. Das Shell-Startskript erstellen (start_lesehilfe.sh)
START_SCRIPT="$PROJECT_DIR/start_lesehilfe.sh"

cat << EOF > "$START_SCRIPT"
#!/bin/bash

# Wechsle in den Projektordner (wichtig für relative Pfade)
cd "$PROJECT_DIR"

# Aktiviere die virtuelle Umgebung und starte das Python-Skript
# Wir verwenden 'exec', um den Prozess zu ersetzen.
source venv/bin/activate
exec python3 Lesehilfe.py
EOF

# Skript ausführbar machen
chmod +x "$START_SCRIPT"

# 4. Die Desktop-Datei erstellen (lesehilfe.desktop)
DESKTOP_FILE="$PROJECT_DIR/lesehilfe.desktop"

cat << EOF > "$DESKTOP_FILE"
[Desktop Entry]
Type=Application
Name=Lesehilfe Start
Comment=Startet das Lesehilfe-Programm mit venv.
# Exec ruft das soeben erstellte Start-Skript auf
Exec=$START_SCRIPT
Terminal=true
Hidden=false
NoDisplay=false
Categories=Utility;
EOF

# 5. Desktop-Datei in den Autostart-Ordner kopieren
AUTOSTART_DIR="$USER_HOME/.config/autostart"

# Sicherstellen, dass der Autostart-Ordner existiert (gehört dem Benutzer)
sudo -u "$USER_NAME" mkdir -p "$AUTOSTART_DIR"

# Datei kopieren und dem Benutzer 'pi' die Rechte geben
cp "$DESKTOP_FILE" "$AUTOSTART_DIR/"
chown "$USER_NAME":"$USER_NAME" "$AUTOSTART_DIR/lesehilfe.desktop"

echo "Installation und Autostart-Konfiguration abgeschlossen."
echo "Bitte starten Sie den Raspberry Pi neu, um das Programm automatisch auszuführen."