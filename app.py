from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set up upload and static folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models for captioning and VQA with error handling
try:
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    use_pipeline = True
    if use_pipeline:
        vqa_pipeline = pipeline("vqa", model="Salesforce/blip-vqa-base")
    else:
        qa_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-large")
        qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-large")
except Exception as e:
    print(f"Error loading models: {e}")

# Helper function for caption generation
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = caption_processor(images=image, return_tensors="pt")
        output = caption_model.generate(**inputs)
        caption = caption_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption."

# Helper functions for question answering
def answer_question_pipeline(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
        result = vqa_pipeline(image=image, question=question)
        return result[0]["answer"] if result else "No answer found."
    except Exception as e:
        print(f"Error answering question with pipeline: {e}")
        return "Error answering question."

def answer_question_model(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = f"Answer the following question based on the image: {question}"
        inputs = qa_processor(images=image, text=prompt, return_tensors="pt")
        output = qa_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        return qa_processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error answering question with model: {e}")
        return "Error answering question."

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/service")
def service():
    return render_template("service.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)

    caption = generate_caption(image_path)

    return jsonify({
        "image_url": f"/static/uploads/{filename}",
        "caption": caption
    })

@app.route("/answer", methods=["POST"])
def answer_image_question():
    if "image" not in request.form or "question" not in request.form:
        return jsonify({"error": "No image or question provided"}), 400

    image_url = request.form["image"]
    question = request.form["question"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_url.split("/")[-1])

    answer = answer_question_pipeline(image_path, question) if use_pipeline else answer_question_model(image_path, question)
    return jsonify({"answer": answer})

# Static file serving for CSS, JS, images
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
