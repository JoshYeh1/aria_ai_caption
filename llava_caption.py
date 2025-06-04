import cv2
import numpy as np
from PIL import Image
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
import argparse
import torch
from ollama import Client
import base64
import io
import time

# Initialize Ollama client once
ollama_client = Client()

torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class StreamingObserver:
    def __init__(self):
        self.last_image = None
        self.caption_ready = True  # Only caption when ready
        self.caption = "Waiting for image..."

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id == aria.CameraId.Rgb and self.caption_ready:
            self.last_image = np.rot90(image, -1)
            self.caption_ready = False  # Block further captioning until done
            self.generate_caption_async(self.last_image)

    def generate_caption_async(self, np_img: np.ndarray):
        pil_img = Image.fromarray(np_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = "Describe the scene."
        response = ollama_client.generate(
            model="llava:latest",
            prompt=prompt,
            images=[img_b64]
        )
        self.caption = response["response"].strip()
        self.caption_ready = True

# Usage (e.g., inside a loop):
try:
    while True:
        if observer.last_image is not None:
            # Original frame (latest stream)
            original = observer.last_image.copy()
            original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

            # Image being captioned
            caption_input = cv2.resize(original_bgr.copy(), (256, 256))

            # Captioned frame
            captioned = original_bgr.copy()
            cv2.putText(
                captioned,
                observer.caption,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Resize for layout
            h, w = 256, 256
            original_resized = cv2.resize(original_bgr, (w, h))
            captioned_resized = cv2.resize(captioned, (w, h))

            # Stack side-by-side
            composite = np.hstack((original_resized, caption_input, captioned_resized))
            cv2.imshow("Live | Caption Input | Captioned", composite)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
