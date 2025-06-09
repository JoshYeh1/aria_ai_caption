from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from ollama import Client

# === Initialize Flask and Ollama ===
app = Flask(__name__)
client = Client()  # Connects to local Ollama server at http://localhost:11434


# === /caption endpoint ===
@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        # Load and preprocess image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((256, 256))  # Resize for LLaVA consistency

        # Convert to base64-encoded PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Generate caption using LLaVA
        response = client.generate(
            model="llava",  # or use "llava-phi3" if you have it installed
            prompt="Describe this image for someone who is visually impaired.",
            images=[image_b64],
        )

        caption = response.get("response", "No caption returned.")
        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === /follow_up endpoint ===
@app.route('/follow_up', methods=['POST'])
def follow_up():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    try:
        # Load and preprocess image
        image_file = request.files['image']
        question = request.form['question']

        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((256, 256))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Ask question with image context
        response = client.generate(
            model="llava",  # or "llava-phi3"
            prompt=question,
            images=[image_b64],
        )

        answer = response.get("response", "No answer returned.")
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === Run server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
