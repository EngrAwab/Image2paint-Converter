#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Updating package lists..."
apt-get update

echo "Installing dependencies..."
# Install jq for JSON parsing, along with other dependencies
apt-get install -y wget unzip fontconfig jq

echo "Downloading Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

echo "Installing Google Chrome..."
# Use dpkg and apt-get to handle dependencies
dpkg -i google-chrome-stable_current_amd64.deb || apt-get -fy install

echo "Determining Chrome version and finding corresponding ChromeDriver version..."
# Dynamically get installed Chrome version
# Note: The command to get the version might need adjustment if google-chrome isn't in the PATH
# This path is standard for Debian installs
CHROME_VERSION=$(/usr/bin/google-chrome --version | cut -d ' ' -f 3 | cut -d '.' -f 1)
# Get the latest known good version for that major release from the JSON endpoint
CHROME_DRIVER_VERSION=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build.json | jq -r ".builds[\"$CHROME_VERSION\"].version")


echo "Downloading ChromeDriver version ${CHROME_DRIVER_VERSION}..."
wget -N https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROME_DRIVER_VERSION}/linux64/chromedriver-linux64.zip -P ~/

echo "Installing ChromeDriver..."
unzip -o ~/chromedriver-linux64.zip -d ~/
rm ~/chromedriver-linux64.zip
# Move to a standard location in the PATH
mv -f ~/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
chmod +x /usr/local/bin/chromedriver

# Install Python dependencies
pip install -r requirements.txt

echo "Build script finished."
