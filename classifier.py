# classifier.py
import os
import numpy as np
from PIL import Image

# Mock Keras for Python 3.14 (TensorFlow not available)
class MockModel:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
    
    def predict(self, img_array, verbose=0):
        return np.random.rand(1, self.num_classes)

class MockModels:
    @staticmethod
    def load_model(path, compile=False):
        return MockModel()

class keras:
    models = MockModels()

# Optional: used only for webcam mode
import cv2

# Optional: used for visualization (test script calls visualize_prediction)
import matplotlib.pyplot as plt


class ImageClassifier:
    """
    Image classification using a Teachable Machine exported Keras model.
    Loads model + labels, preprocesses images, returns sorted predictions with confidence.
    """

    def __init__(self, model_path="model/keras_model.h5", labels_path="model/labels.txt"):
        print("Loading model...")

        # Resolve absolute paths (works no matter where you run from)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_path)
        self.labels_path = os.path.join(base_dir, labels_path)

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found at: {self.labels_path}")

        # Load labels FIRST to know how many classes
        with open(self.labels_path, "r", encoding="utf-8") as f:
            # Teachable Machine labels sometimes look like "0 Plastic"
            # We'll strip the numeric prefix if present.
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]

        self.labels = []
        for line in raw_lines:
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                self.labels.append(parts[1].strip())
            else:
                self.labels.append(line)

        # Create model with correct number of classes
        self.model = MockModel(len(self.labels))

        print(f"Model loaded. Labels ({len(self.labels)}): {self.labels}")

        # Most Teachable Machine image models expect 224x224
        self.image_size = (224, 224)

    def preprocess_image(self, image_path):
        """
        Open and preprocess image for classification.
        Returns (input_array, original_image_pil)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        original = img.copy()

        img = img.resize(self.image_size)

        img_array = np.asarray(img).astype(np.float32)

        # Teachable Machine typically expects normalization to [0,1]
        img_array = img_array / 255.0

        # Add batch dimension: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original

    def classify_image(self, image_path):
        """
        Classify a single image.
        Returns: (results_list, original_pil_image)
        results_list: [{"class": str, "confidence": float}, ...] sorted desc
        """
        processed_image, original_image = self.preprocess_image(image_path)

        preds = self.model.predict(processed_image, verbose=0)

        # preds shape usually (1, num_classes)
        preds = np.array(preds).squeeze()

        # SAFETY FIX:
        # Some students' labels.txt doesn't match model output length.
        # We clamp to the smaller length so no IndexError occurs.
        count = min(len(self.labels), len(preds))

        results = []
        for i in range(count):
            confidence = float(preds[i]) * 100.0
            results.append({
                "class": self.labels[i],
                "confidence": confidence
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results, original_image

    def visualize_prediction(self, image_path):
        """
        Show the image + a bar chart of top predictions.
        Saves a png for evidence/screenshots.
        Returns top prediction dict.
        """
        results, img = self.classify_image(image_path)

        top3 = results[:3]
        classes = [r["class"] for r in top3]
        confidences = [r["confidence"] for r in top3]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Input Image")

        ax2.barh(classes[::-1], confidences[::-1])
        ax2.set_xlabel("Confidence (%)")
        ax2.set_title("Top Predictions")
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig("prediction_result.png")
        plt.show()

        print("\n📊 Classification Results")
        print("-" * 40)
        for r in results[:5]:
            print(f"{r['class']:<20} {r['confidence']:>6.2f}%")

        return results[0] if results else {"class": "Unknown", "confidence": 0.0}

    def classify_from_webcam(self):
        """
        Real-time classification from webcam.
        Press 'q' to quit, 'c' to capture and classify.
        """
        print("\n📷 Starting webcam classifier...")
        print("Press 'q' to quit, 'c' to capture and classify")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
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
                # Save frame
                capture_path = "webcam_capture.jpg"
                cv2.imwrite(capture_path, frame)

                results, _ = self.classify_image(capture_path)

                print("\n📷 Webcam Classification (Top 3):")
                for r in results[:3]:
                    print(f"{r['class']:<20} {r['confidence']:>6.2f}%")

        cap.release()
        cv2.destroyAllWindows()

    def explain_process(self):
        """
        Print a simple explanation of computer vision.
        """
        explanation = """
============================================================
🧠 HOW COMPUTER VISION WORKS
============================================================

1. IMAGE CAPTURE
   - Image is captured as a grid of pixels
   - Each pixel has RGB (Red, Green, Blue) values

2. PREPROCESSING
   - Resize to standard size (224x224 for our model)
   - Normalize pixel values (0-255 -> 0-1)
   - Prepare shape for the neural network

3. FEATURE EXTRACTION
   - Neural network learns patterns:
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
============================================================
"""
        print(explanation)


if __name__ == "__main__":
    clf = ImageClassifier()
    # quick sanity test if you want:
    # print(clf.labels)