# Week 5 Project: Smart Image Classifier

## Overview
This project builds an image classification system using **Google Teachable Machine** and **Python**.  
It can classify objects from uploaded images and from a live webcam feed.

## Project Structure
- `classifier.py` - Loads model, preprocesses images, predicts classes, visualizes results, webcam mode
- `web_interface.py` - Flask web app for uploading images and showing predictions
- `test_classifier.py` - Menu-based tester for folder images, webcam, and explanation
- `model/` - Teachable Machine exported model (`keras_model.h5`) and `labels.txt`

## Setup
```bash
pip install -r requirements.txt