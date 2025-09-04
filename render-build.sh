#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Installa le librerie Python
pip install -r requirements.txt

# 2. Installa ffmpeg
apt-get update && apt-get install -y ffmpeg
