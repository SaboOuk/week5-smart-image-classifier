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

##Challenges Encountered
One challenge was configuring the local Python environment on Windows, as newer Python versions caused compatibility issues with TensorFlow and OpenCV. This was addressed by targeting Python 3.10/3.11, which are supported by these libraries.

##Real-World Applications
Medical: Assisting with analysis of medical images such as X-rays or scans.
Security: Object and facial recognition in monitoring systems.
Retail: Automated product recognition and inventory tracking.

##What I Learned
I learned how images are represented as pixel data and processed by neural networks. I gained experience with preprocessing steps such as resizing and normalization, as well as understanding prediction confidence scores. This project helped me understand how computer vision models are integrated into Python applications.