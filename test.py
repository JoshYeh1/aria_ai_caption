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

# === Initialize Ollama client ===
print("🔄 Connecting to Ollama...")
client = Client()
print("✅ LLaVA (Ollama) client initialized.")

# === Streaming Observer Class ===
class StreamingObserver:
    def __init__(self):
        self.pending_frame = None  # New frame to caption
        self.has_new_frame = False

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id != aria.CameraId.Rgb:
            return
        if not self.has_new_frame:
            self.pending_frame = np.rot90(image, -1).copy()
            self.has_new_frame = True

    def generate_caption(self, np_img: np.ndarray) -> str:
        start_time = time.time()
        try:
            image = Image.fromarray(np_img).convert("RGB").resize((256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = client.generate(
                model="llava",
                prompt="Describe this scene for someone who is visually impaired.",
                images=[image_b64],
            )
            print(f"Duration: {time.time()-start_time:.2f} seconds")
            return response.get("response", "No caption returned.")
        except Exception as e:
            return f"Exception: {e}"


# === CLI Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--interface",
    type=str,
    required=True,
    choices=["usb", "wifi"],
    help="Connection type: usb or wifi",
)
parser.add_argument("--debug",action="store_true",help="Enable debug output")
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
    print("📡 Streaming started over Wi-Fi.")

# === Aria Streaming Setup ===
print("🔌 Initializing Aria streaming client...")
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
print("✅ Connected to Aria. Streaming started.")

# === OpenCV Display Loop ===
cv2.namedWindow("Aria RGB + LLaVA Caption", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aria RGB + LLaVA Caption", 640, 480)

try:
    while True:
        if observer.has_new_frame:
            # Get and reset pending frame
            frame_rgb = observer.pending_frame
            observer.has_new_frame = False

            # Generate caption
            image = Image.fromarray(frame_rgb).convert("RGB").resize((256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            start_time = time.time()
            response = client.generate(
                model="llava",
                prompt="Describe this scene for someone who is visually impaired.",
                images=[image_b64],
            )
            caption = response.get("response", "No caption returned.")
            print(f"🕒 Caption duration: {time.time() - start_time:.2f}s")

            # Display captioned frame
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame_bgr,
                caption,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Aria RGB + LLaVA Caption", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    streaming_client.unsubscribe()
    cv2.destroyAllWindows()

