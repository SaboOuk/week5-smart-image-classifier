# classifier.py
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


class ImageClassifier:
    """
    Image classification using computer vision
    Demonstrates concepts from Chapter 24
    """

    def __init__(self, model_path="model/keras_model.h5", labels_path="model/labels.txt"):
        """Load the trained model and labels"""
        print("Loading model...")
        self.model = keras.models.load_model(model_path, compile=False)

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Model expects 224x224 images (Teachable Machine default)
        self.image_size = (224, 224)

        print(f"Model loaded with {len(self.labels)} classes: {self.labels}")

    def preprocess_image(self, image_path: str):
        """
        Prepare image for classification:
        - RGB conversion
        - resize to 224x224
        - normalize to 0..1
        - add batch dimension
        """
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(self.image_size)

        img_array = np.array(img_resized).astype("float32")
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, img  # return original PIL image too

    def classify_image(self, image_path: str):
        """Classify a single image and return sorted results."""
        processed_image, original_image = self.preprocess_image(image_path)

        predictions = self.model.predict(processed_image, verbose=0)[0]

        results = []
        for i, label in enumerate(self.labels):
            confidence = float(predictions[i]) * 100
            results.append({"class": label, "confidence": confidence})

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results, original_image

    def visualize_prediction(self, image_path: str):
        """Show image + bar chart of top predictions and save result."""
        results, img = self.classify_image(image_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: image
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Input Image")

        # Right: top 3 predictions bar chart
        top = results[:3]
        classes = [r["class"] for r in top]
        confs = [r["confidence"] for r in top]

        ax2.barh(classes[::-1], confs[::-1])
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Confidence (%)")
        ax2.set_title("Top 3 Predictions")

        for i, c in enumerate(confs[::-1]):
            ax2.text(c + 1, i, f"{c:.1f}%")

        plt.tight_layout()
        plt.savefig("prediction_result.png")
        print("🖼 Saved visualization as 'prediction_result.png'")
        plt.show()

        print("\n📌 Classification Results")
        print("-" * 40)
        for r in results[:5]:
            print(f"{r['class']:<20} {r['confidence']:>6.2f}%")

        return results[0]  # top prediction

    def classify_from_webcam(self):
        """Real-time classification from webcam. Press 'c' to capture, 'q' to quit."""
        print("\n📷 Starting webcam classifier...")
        print("Press 'c' to capture and classify, 'q' to quit")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Webcam - Press C to Classify, Q to Quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                capture_path = "webcam_capture.jpg"
                cv2.imwrite(capture_path, frame)
                results, _ = self.classify_image(capture_path)

                print("\n📷 Webcam Classification (Top 3)")
                for r in results[:3]:
                    print(f"{r['class']:<20} {r['confidence']:>6.2f}%")

        cap.release()
        cv2.destroyAllWindows()

    def explain_process(self):
        """Explain how computer vision works."""
        explanation = """
🧠 HOW COMPUTER VISION WORKS
============================================================

1. IMAGE CAPTURE
- Image is captured as a grid of pixels
- Each pixel has RGB (Red, Green, Blue) values

2. PREPROCESSING
- Resize to standard size (224x224 for our model)
- Normalize pixel values (0-255 -> 0-1)
- Prepare shape for neural network (add batch dimension)

3. FEATURE EXTRACTION
- Neural network finds patterns:
  * Edges and corners (early layers)
  * Shapes and textures (middle layers)
  * Objects and concepts (deep layers)

4. CLASSIFICATION
- Final layer outputs probability for each class
- Highest probability is the prediction

5. CHALLENGES
- Lighting changes appearance
- Different angles look different
- Partial occlusion (things blocking view)
- Similar looking objects
"""
        print(explanation)
