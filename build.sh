#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setting up a local Chrome and ChromeDriver..."

# Define a local directory for the binaries
BIN_DIR=$(pwd)/.local-bin
mkdir -p $BIN_DIR

# --- Google Chrome ---
echo "Downloading Google Chrome..."
wget -P /tmp https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

echo "Extracting Google Chrome..."
# dpkg-deb is a safe way to extract without installing system-wide
dpkg-deb -x /tmp/google-chrome-stable_current_amd64.deb $BIN_DIR/chrome-unpacked
rm /tmp/google-chrome-stable_current_amd64.deb

# --- ChromeDriver ---
echo "Downloading ChromeDriver..."
# Find the latest stable version
CHROME_DRIVER_VERSION=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)
wget -N https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROME_DRIVER_VERSION}/linux64/chromedriver-linux64.zip -P /tmp

echo "Extracting ChromeDriver..."
unzip -o /tmp/chromedriver-linux64.zip -d $BIN_DIR
rm /tmp/chromedriver-linux64.zip

# Move chromedriver to a simpler path and make it executable
mv $BIN_DIR/chromedriver-linux64/chromedriver $BIN_DIR/chromedriver
rm -r $BIN_DIR/chromedriver-linux64
chmod +x $BIN_DIR/chromedriver

echo "Build script finished successfully."
