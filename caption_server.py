# This runs GPU server connection 
# Make sure this is running before running programs needing server requests.
# IMPORTANT: Change your server address 
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from ollama import Client

# === Initialize Flask and Ollama ===
app = Flask(__name__)
client = Client()  #connects to local Ollama server at http://localhost:11434
# can use "llava" or "llava-phi3" or "deepseek-v2:16b"
ai_model = f"llava"

# === Prompts ===
CAPTION_PROMPT = (
    "Caption the image concisely for someone who is visually impaired. "
    "Only mention observable objects and potential hazards. "
    "Keep it under two sentences."
)

FOLLOW_UP_PROMPT_TEMPLATE = (
    "Answer concisely in less than two sentences: {}"
)

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
            model=ai_model,  # can use "llava" or "llava-phi3" or "deepseek-v2:16b"
            prompt=CAPTION_PROMPT,
            images=[image_b64],
        )

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

        prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(question)

        response = client.generate(
            model=ai_model,  # use "llava" or "llava-phi3"
            prompt=prompt,
            images=[image_b64],
        )

        answer = response.get("response", "No answer returned.")
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === Run server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded = True)

