from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from ollama import Client

# === Initialize Flask and Ollama ===
app = Flask(__name__)
client = Client()  #connects to local Ollama server at http://localhost:11434

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        #load and preprocess image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((256, 256))  #resize for faster analysis

        #conversts to base64 png
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        #generate caption
        response = client.generate(
            model="llava-phi3",  # can use "llava" or "llava-phi3"
            prompt="Describe this image concisely including objects and hazards. Do not make assumptions beyond observable facial expressions.",
            images=[image_b64],
        )
        #Alternative Promt:Describe this image briefly for someone who is visually impaired. Exclude assumptions except for facial expressions
        caption = response.get("response", "No caption returned.")
        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/follow_up', methods=['POST'])
def follow_up():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    try:
        image_file = request.files['image']
        question = request.form['question']

        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((256, 256))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        response = client.generate(
            model="llava",  # use "llava" or "llava-phi3"
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

