import os

import matplotlib.pyplot as plt
import numpy as np

log_dir = "data/20251129/40cm/air/"
# list all log files in the directory

log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]
log_file = log_files[1]

# Parse the log file
photon_counts = []

with open(log_file, "r") as f:
    for line in f:
        # Extract the DATA section
        if "DATA:" in line:
            data_part = line.split("DATA:")[1].strip()
            # Split hex bytes
            hex_bytes = data_part.split()

            # Convert pairs of hex bytes to 16-bit values (little-endian)
            for i in range(0, len(hex_bytes), 2):
                if i + 1 < len(hex_bytes):
                    low_byte = int(hex_bytes[i], 16)
                    high_byte = int(hex_bytes[i + 1], 16)
                    count = low_byte | (high_byte << 8)
                    photon_counts.append(count)

photon_counts = np.array(photon_counts)

# Create histogram plot
plt.figure(figsize=(12, 6))
plt.plot(photon_counts, linewidth=0.5, color="darkblue", alpha=0.7)
plt.xlabel("Bin Index", fontsize=12)
plt.ylabel("Photon Counts", fontsize=12)
plt.title(f"LiDAR Raw Histogram - {len(photon_counts)} bins", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig(log_file.replace(".log", ".png"))

print(f"Total bins: {len(photon_counts)}")
print(f"Count range: {photon_counts.min()} - {photon_counts.max()}")
print(f"Mean count: {photon_counts.mean():.2f}")

# print(f"Peak bin: {photon_counts.argmax()} with count {photon_counts.max()}")
peak_indices = []

# Gradient-Based
# BIAS = 2
# GRAD_TH = 5 * photon_counts.mean()
# for idx, pc in enumerate(photon_counts):
#     if idx == 0:
#         continue
#     gradient = pc - photon_counts[idx - 1]
#     if gradient > GRAD_TH:
#         peak_indices.append(idx)


# Sliding Window Max
if "air" in log_file:
    BIAS = 2.05
    speed_of_light = 299792458  # meters per second
elif "water" in log_file:
    BIAS = 1.52
    speed_of_light = 299792458 / 1.33  # meters per second


SW_SIZE = 1600
intervals = []
for sw_idx in range(photon_counts.size // SW_SIZE):
    peak_indices.append(
        photon_counts[sw_idx * SW_SIZE : (sw_idx + 1) * SW_SIZE].argmax()
        # + sw_idx * SW_SIZE
    )
    if sw_idx > 0:
        intervals.append(peak_indices[-1] - peak_indices[-2])

print(f"Avg Inter-Peak bins: {np.mean(intervals):.2f}")

print(peak_indices)

# Calculate distance using first peak
if peak_indices:
    first_peak_bin = np.mean(peak_indices)
    bin_resolution_ps = 104.17  # picoseconds per bin

    # Convert bin index to time (ps to seconds)
    time_of_flight_s = first_peak_bin * bin_resolution_ps * 1e-12

    # Distance = (speed of light * time of flight) / 2
    # Divide by 2 because light travels to target and back
    distance_m = (speed_of_light * time_of_flight_s) / 2 - BIAS

    print(f"\nFirst peak at bin: {first_peak_bin}")
    print(f"Time of flight: {time_of_flight_s * 1e9:.2f} ns")
    print(f"Distance: {distance_m:.2f} m")
