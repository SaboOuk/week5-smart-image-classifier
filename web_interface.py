# web_interface.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from classifier import ImageClassifier

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

classifier = ImageClassifier()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>🖼 AI Image Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;
           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .container { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
    h1 { text-align: center; color: #333; }
    .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 30px; text-align: center; margin: 20px 0; }
    button { background: #667eea; color: white; border: none; padding: 10px 30px; border-radius: 5px; cursor: pointer; font-size: 16px; }
    button:hover { background: #764ba2; }
    #results { margin-top: 30px; padding: 20px; background: #f8f8f8; border-radius: 5px; display: none; }
    .result-bar { background: #667eea; color: white; padding: 10px; margin: 5px 0; border-radius: 5px; }
    #preview { max-width: 300px; margin: 20px auto; display: none; }
    img { max-width: 100%; border-radius: 5px; }
  </style>
</head>
<body>
<div class="container">
  <h1>🖼 AI Image Classifier</h1>
  <p style="text-align:center;">Upload an image to see what the AI thinks it is!</p>

  <div class="upload-area">
    <p>📁 Choose an image file</p>
    <input type="file" id="fileInput" accept="image/*" onchange="previewImage()">
    <div id="preview"></div>
    <br><br>
    <button onclick="classifyImage()">🔍 Classify Image</button>
  </div>

  <div id="results"></div>

  <div style="margin-top:30px; padding:20px; background:#f0f0f0; border-radius:5px;">
    <h3>ℹ About This Classifier</h3>
    <p>This AI was trained using Teachable Machine to recognize: <strong>{{ classes }}</strong></p>
    <p>The model uses a neural network to identify patterns in images, similar to how your brain recognizes objects!</p>
  </div>
</div>

<script>
function previewImage(){
  const file = document.getElementById('fileInput').files[0];
  const preview = document.getElementById('preview');
  if(file){
    const reader = new FileReader();
    reader.onload = function(e){
      preview.innerHTML = '<img src="' + e.target.result + '">';
      preview.style.display = 'block';
    }
    reader.readAsDataURL(file);
  }
}

function classifyImage(){
  const file = document.getElementById('fileInput').files[0];
  if(!file){
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  fetch('/classify', { method:'POST', body: formData })
    .then(response => response.json())
    .then(data => showResults(data))
    .catch(error => alert("Error: " + error));
}

function showResults(data){
  const resultsDiv = document.getElementById('results');
  let html = "<h3>🎯 Classification Results</h3>";

  data.results.forEach(result => {
    const width = result.confidence;
    html += '<div class="result-bar" style="width:' + width + '%">';
    html += result.class + ": " + result.confidence.toFixed(1) + "%";
    html += "</div>";
  });

  resultsDiv.innerHTML = html;
  resultsDiv.style.display = "block";
}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    classes = ", ".join(classifier.labels)
    return HTML_TEMPLATE.replace("{{ classes }}", classes)

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    results, _ = classifier.classify_image(filepath)

    return jsonify({
        "filename": filename,
        "results": results[:3]  # top 3
    })

if __name__ == "__main__":
    print("🚀 Starting Image Classification Server...")
    print("📍 Visit http://localhost:5000 to upload images")
    app.run(debug=True, port=5000)
