import asyncio
import gc
import os
import struct
import sys
from collections import deque

import numpy as np
import reflex as rx
import serial
import serial.tools.list_ports
import torch

# Add parent directory to path to allow importing train.py and eval_real.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_real import (
    build_model,
    extract_distance_from_prediction,
    preprocess_real_data,
)

# Import from project files
# We assume these files are in the same directory
from train import (
    Longformer1D,
    MobileNet1D,
    Performer1D,
    ResNet1D,
    Transformer1D,
    UNet1D,
)

# Constants
PERIOD_BINS = 1600

# Global state storage to avoid pickling issues with Reflex
global_ser = None
global_model = None
global_device = None

class State(rx.State):
    # Connection Settings
    port: str = ""
    available_ports: list[str] = []
    baud_rate: int = 1000000
    is_connected: bool = False
    status: str = "Disconnected"

    # Model Settings
    models: list[str] = ["unet", "mcd_unet", "performer", "mobilenet", "resnet", "transformer", "longformer"]
    selected_model: str = "mcd_unet"

    # Data for Charts
    # Format: [{"bin": 0, "raw": 0.5, "pred": 0.1}, ...]
    chart_data: list[dict] = []

    # Metrics
    peak_bin: int = 0
    distance_m: float = 0.0
    snr: float = 0.0
    confidence: float = 0.0
    frame_count: int = 0

    # Internal
    _running: bool = False
    _photon_buffer: list = []

    def on_load(self):
        global global_device
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refresh_ports()
        self.load_model()

    def refresh_ports(self):
        self.available_ports = [p.device for p in serial.tools.list_ports.comports()]
        if self.available_ports and not self.port:
            self.port = self.available_ports[0]

    def set_model(self, model_name: str):
        self.selected_model = model_name
        self.load_model()

    def load_model(self):
        global global_model, global_device
        try:
            global_model = build_model(self.selected_model)
            if global_model:
                # Load checkpoint
                ckpt_path = f"checkpoint_{self.selected_model}.pth"
                if os.path.exists(ckpt_path):
                    checkpoint = torch.load(ckpt_path, map_location=global_device)
                    # Handle different checkpoint formats if necessary
                    if 'model_state_dict' in checkpoint:
                        global_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        global_model.load_state_dict(checkpoint)
                    global_model.to(global_device)
                    global_model.eval()
                    self.status = f"Loaded {self.selected_model}"
                else:
                    self.status = f"Checkpoint {ckpt_path} not found"
        except Exception as e:
            self.status = f"Error loading model: {str(e)}"

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            return self.connect()

    def disconnect(self):
        global global_ser
        self._running = False
        if global_ser and global_ser.is_open:
            global_ser.close()
        self.is_connected = False
        self.status = "Disconnected"

    def connect(self):
        global global_ser
        try:
            if not self.port:
                self.status = "No port selected"
                return

            global_ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            self.is_connected = True
            self._running = True
            self.status = f"Connected to {self.port}"
            return State.read_loop
        except Exception as e:
            self.status = f"Connection failed: {str(e)}"
            self.is_connected = False

    async def read_loop(self):
        global global_ser
        # Buffer for incoming bins
        # Note: In this recursive pattern, local variables like bin_buffer are lost between calls
        # We need to store bin_buffer in self or pass it around.
        # Storing in self is easier.

        if not self._running or not self.is_connected:
            return

        try:
            # Non-blocking read check
            if global_ser and global_ser.in_waiting > 0:
                # Read header
                b1 = global_ser.read(1)
                if b1 == b'\xaa':
                    b2 = global_ser.read(1)

                    if b2 == b'\xf1': # Histogram Data
                        # Read Length (2 bytes)
                        len_bytes = global_ser.read(2)
                        if len(len_bytes) == 2:
                            payload_len = (len_bytes[0] << 8) | len_bytes[1]

                            # Read Payload + CRC
                            payload = global_ser.read(payload_len)
                            crc_byte = global_ser.read(1)

                            if len(payload) == payload_len:
                                # Payload = Index(1) + StartPos(4) + RawData(...)
                                # idx = payload[0]
                                raw_vals = payload[5:]

                                # Convert bytes to 16-bit ints (Little Endian)
                                # raw_vals is a byte string
                                # We can use struct or numpy
                                # Assuming raw_vals is a sequence of uint16

                                # Parse into integers
                                new_bins = []
                                for i in range(0, len(raw_vals), 2):
                                    if i + 1 < len(raw_vals):
                                        val = raw_vals[i] | (raw_vals[i+1] << 8)
                                        new_bins.append(val)

                                self._photon_buffer.extend(new_bins)

                                # Safety check for buffer size
                                if len(self._photon_buffer) > PERIOD_BINS * 10:
                                    self._photon_buffer = []
                                    print("Buffer overflow, clearing buffer")

                                # Check if we have enough for a full period
                                if len(self._photon_buffer) >= PERIOD_BINS:
                                    # Take one period
                                    current_period = self._photon_buffer[:PERIOD_BINS]
                                    self._photon_buffer = self._photon_buffer[PERIOD_BINS:] # Keep remainder

                                    # Process this period
                                    await self.process_frame(current_period)

                    elif b2 == b'\x0f': # Distance Data (Optional, we calculate our own)
                        global_ser.read(8) # Consume

        except Exception as e:
            print(f"Serial Error: {e}")

        # Yield to event loop and schedule next run
        await asyncio.sleep(0.001)
        if self._running:
            return State.read_loop

    async def process_frame(self, photon_counts):
        global global_model, global_device
        # Run inference
        try:
            counts_array = np.array(photon_counts)

            # Preprocess
            # preprocess_real_data expects numpy array
            input_tensor = preprocess_real_data(counts_array).to(global_device)

            # Inference
            if global_model:
                with torch.no_grad():
                    # Match eval_real.py: direct output, no extra sigmoid unless model has it
                    prediction = global_model(input_tensor).cpu().numpy()[0, 0]
            else:
                prediction = np.zeros_like(counts_array, dtype=float)

            # Extract Metrics
            metrics = extract_distance_from_prediction(
                prediction,
                speed_of_light=299792458 / 1.33, # Water
                distance_bias=1.52
            )

            # Update State
            # async with self: # Removed for compatibility
            self.peak_bin = int(metrics["peak_bin"])
            self.distance_m = float(metrics["distance_m"])
            self.snr = float(metrics["snr"])
            self.confidence = float(metrics["peak_value"])

            # Prepare chart data (downsample if needed for performance)
            # Sending 1600 points might be heavy for every frame?
            # Reflex/Recharts can handle it, but maybe limit update rate?
            # For now, update every frame.

            # Normalize raw counts for display
            max_val = np.max(counts_array)
            norm_counts = counts_array / max_val if max_val > 0 else counts_array

            self.chart_data = [
                {
                    "bin": i,
                    "raw": float(norm_counts[i]),
                    "pred": float(prediction[i])
                }
                for i in range(0, len(counts_array), 4) # Downsample by 4 for UI performance
            ]

            # Periodic cleanup
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Processing Error: {e}")


def index():
    return rx.container(
        rx.vstack(
            rx.heading("Underwater LiDAR Live Inference", size="8"),

            # Controls
            rx.hstack(
                rx.select(
                    State.available_ports,
                    placeholder="Select Port",
                    on_change=State.set_port,
                    value=State.port
                ),
                rx.button(
                    "Refresh Ports",
                    on_click=State.refresh_ports
                ),
                rx.select(
                    State.models,
                    value=State.selected_model,
                    on_change=State.set_model
                ),
                rx.button(
                    rx.cond(State.is_connected, "Disconnect", "Connect"),
                    on_click=State.toggle_connection,
                    color_scheme=rx.cond(State.is_connected, "red", "green")
                ),
                rx.text(State.status, color="gray"),
                spacing="4",
                align="center"
            ),

            rx.divider(),

            # Metrics
            rx.hstack(
                rx.vstack(
                    rx.text("Distance", font_size="sm", font_weight="medium", color="gray"),
                    rx.heading(f"{State.distance_m * 100:.3f} cm", size="6"),
                    rx.text("Estimated", font_size="xs", color="gray"),
                    align="center",
                    padding="4",
                    # border="1px solid #eaeaea",
                    # border_radius="md",
                ),
                rx.vstack(
                    rx.text("Peak Bin", font_size="sm", font_weight="medium", color="gray"),
                    rx.heading(f"{State.peak_bin}", size="6"),
                    align="center",
                    padding="4",
                    # border="1px solid #eaeaea",
                    # border_radius="md",
                ),
                rx.vstack(
                    rx.text("Confidence", font_size="sm", font_weight="medium", color="gray"),
                    rx.heading(f"{State.confidence:.2f}", size="6"),
                    align="center",
                    padding="4",
                    # border="1px solid #eaeaea",
                    # border_radius="md",
                ),
                rx.vstack(
                    rx.text("SNR", font_size="sm", font_weight="medium", color="gray"),
                    rx.heading(f"{State.snr:.2f}", size="6"),
                    align="center",
                    padding="4",
                    # border="1px solid #eaeaea",
                    # border_radius="md",
                ),
                spacing="6",
                width="100%",
                justify="center"
            ),

            # Charts
            rx.box(
                rx.recharts.composed_chart(
                    rx.recharts.area(
                        data_key="raw",
                        stroke="#8884d8",
                        fill="#8884d8",
                        name="Raw Histogram (Norm)"
                    ),
                    rx.recharts.line(
                        data_key="pred",
                        stroke="#ff7300",
                        stroke_width=2,
                        dot=False,
                        name="Prediction"
                    ),
                    rx.recharts.x_axis(data_key="bin"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.legend(),
                    data=State.chart_data,
                    height=400,
                    width="100%",
                ),
                width="100%",
                padding="4"
            ),

            spacing="6",
            padding="6",
            width="100%",
            max_width="1200px"
        )
    )

app = rx.App()
app.add_page(index, on_load=State.on_load)
