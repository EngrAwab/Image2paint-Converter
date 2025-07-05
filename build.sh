#!/usr/bin/env bash
# exit on error
set -o errexit

# Add sudo to package management commands
echo "Updating package lists..."
sudo apt-get update

echo "Installing dependencies..."
sudo apt-get install -y wget unzip fontconfig

echo "Downloading Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

echo "Installing Google Chrome..."
# Use dpkg and apt-get to handle dependencies with sudo
sudo dpkg -i google-chrome-stable_current_amd64.deb || sudo apt-get -fy install

echo "Determining Chrome version and finding corresponding ChromeDriver version..."
# Dynamically get installed Chrome version
CHROME_VERSION=$(google-chrome --version | cut -d ' ' -f 3 | cut -d '.' -f 1)
# Get the latest known good version for that major release
CHROME_DRIVER_VERSION=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build.json | jq -r ".builds[\"$CHROME_VERSION\"].version")

echo "Downloading ChromeDriver version ${CHROME_DRIVER_VERSION}..."
wget -N https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROME_DRIVER_VERSION}/linux64/chromedriver-linux64.zip -P ~/

echo "Installing ChromeDriver..."
unzip -o ~/chromedriver-linux64.zip -d ~/
rm ~/chromedriver-linux64.zip
# Move to a standard location in the PATH
sudo mv -f ~/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
sudo chmod +x /usr/local/bin/chromedriver

# Install Python dependencies
pip install -r requirements.txt

echo "Build script finished."
