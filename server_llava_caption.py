import cv2
import numpy as np
import time
from PIL import Image
from ollama import Client
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
import argparse
import sys
import torch
import base64
import io
import threading
import requests

# === Initialize Ollama client ===
print("Connecting to Ollama...")
client = Client()
print("LLaVA (Ollama) client initialized.")

# === Streaming Observer Class ===
class StreamingObserver:
    def __init__(self):
        self.last_image = None
        self.last_caption_time = 0
        self.cooldown = 1.5  # seconds between captions
        self.caption = "Waiting for image..."
        self.caption_in_progress = False

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id == aria.CameraId.Rgb:
            self.last_image = np.rot90(image, -1)
            self.maybe_caption()

    def maybe_caption(self):
        now = time.time()
        if (
            self.last_image is not None
            and not self.caption_in_progress
            and now - self.last_caption_time >= self.cooldown
        ):
            print("Triggering captioning...")
            self.caption_in_progress = True
            self.last_caption_time = now
            threading.Thread(
                target=self._caption_worker, args=(self.last_image.copy(),)
            ).start()

    def _caption_worker(self, image):
        try:
            start = time.time()
            caption = self.generate_caption(image)
            duration = time.time() - start

            self.caption = caption
            print("Caption from LLaVA:", caption)
            print(f"Caption generation took {duration:.2f} seconds")
        finally:
            self.caption_in_progress = False

    def generate_caption(self, np_img: np.ndarray) -> str:
        try:
            # Convert NumPy image to PIL and encode as PNG in-memory
            image = Image.fromarray(np_img).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Send image to Flask caption server
            files = {'image': ('frame.png', buffer, 'image/png')}
            response = requests.post("http://10.100.241.227:8000/caption", files=files)

            if response.status_code == 200:
                return response.json().get("caption", "No caption received.")
            else:
                return f"Server error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Exception during captioning: {e}"

# === CLI Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--interface",
    type=str,
    required=True,
    choices=["usb", "wifi"],
    help="Connection type: usb or wifi",
)
args = parser.parse_args()

# === Optional WiFi Device Setup ===
if args.interface == "wifi":
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)
    device = device_client.connect()
    streaming_manager = device.streaming_manager

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = "profile18"
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    streaming_manager.start_streaming()
    print("Streaming started over Wi-Fi.")

# === Aria Streaming Setup ===
print("Initializing Aria streaming client...")
aria.set_log_level(aria.Level.Info)
streaming_client = aria.StreamingClient()

config = streaming_client.subscription_config
config.subscriber_data_type = aria.StreamingDataType.Rgb
config.message_queue_size[aria.StreamingDataType.Rgb] = 1

options = aria.StreamingSecurityOptions()
options.use_ephemeral_certs = True
config.security_options = options

streaming_client.subscription_config = config

# === Observer & Streaming Start ===
observer = StreamingObserver()
streaming_client.set_streaming_client_observer(observer)
streaming_client.subscribe()
print("Connected to Aria. Streaming started.")

# === OpenCV Display Loop ===
cv2.namedWindow("Aria RGB + LLaVA Caption", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aria RGB + LLaVA Caption", 640, 480)

try:
    while True:
        if observer.last_image is not None:
            frame = cv2.cvtColor(observer.last_image, cv2.COLOR_RGB2BGR)
            # Draw caption
            cv2.putText(
                frame,
                observer.caption,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Aria RGB + LLaVA Caption", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        time.sleep(0.01)  # Sleep for 10ms

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    streaming_client.unsubscribe()
    cv2.destroyAllWindows()
    print("Exiting.")
