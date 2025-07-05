#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# --- Install Google Chrome and ChromeDriver ---
echo "Updating package lists..."
apt-get update

echo "Installing dependencies..."
apt-get install -y wget unzip fontconfig

echo "Downloading Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

echo "Installing Google Chrome..."
# Use dpkg and apt-get to handle dependencies
dpkg -i google-chrome-stable_current_amd64.deb || apt-get -fy install

echo "Downloading ChromeDriver..."
# You may need to update this URL if a new version of Chrome is released
# Check for the latest stable version here: https://googlechromelabs.github.io/chrome-for-testing/
CHROME_DRIVER_VERSION=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)
wget -N https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROME_DRIVER_VERSION}/linux64/chromedriver-linux64.zip -P ~/

echo "Installing ChromeDriver..."
unzip -o ~/chromedriver-linux64.zip -d ~/
rm ~/chromedriver-linux64.zip
mv -f ~/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
chmod +x /usr/local/bin/chromedriver

echo "Build script finished."
