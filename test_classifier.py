# test_classifier.py
from classifier import ImageClassifier
import os

def main():
    print("=" * 60)
    print("🧪 IMAGE CLASSIFIER TEST")
    print("=" * 60)

    classifier = ImageClassifier()

    classifier.explain_process()

    while True:
        print("\n" + "=" * 60)
        print("TESTING OPTIONS")
        print("=" * 60)
        print("1. Classify images from test_images folder")
        print("2. Use webcam for real-time classification")
        print("3. Start web interface")
        print("4. Learn about computer vision challenges")
        print("5. Quit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            folder = "test_images"
            if not os.path.exists(folder):
                print("No test_images folder found.")
                continue

            images = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg",".webp"))]
            if not images:
                print("No images found in test_images/. Add some images and try again.")
                continue

            for img in images:
                path = os.path.join(folder, img)
                print(f"\n📷 Testing: {img}")
                top = classifier.visualize_prediction(path)
                print(f"✅ Top prediction: {top['class']} ({top['confidence']:.1f}%)")

        elif choice == "2":
            classifier.classify_from_webcam()

        elif choice == "3":
            print("Run this in a separate terminal:")
            print("python web_interface.py")

        elif choice == "4":
            print("\nCommon challenges:")
            print("- Lighting changes")
            print("- Different angles")
            print("- Occlusion (blocked view)")
            print("- Similar objects")
            print("- Background clutter")

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
