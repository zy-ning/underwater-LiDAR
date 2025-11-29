import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import erf
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# PART 1: The Simulator (Data Generator)
# ==========================================


class UnderwaterSPADSimulator:
    def __init__(
        self, bin_resolution_ps=104.17, lidar_period_bins=1600, laser_pulse_fwhm_ps=200
    ):
        """
        Initializes the simulator with SPAD hardware constraints.

        Args:
            bin_resolution_ps: Time resolution of one bin in picoseconds.
            lidar_period_bins: Total number of bins in the histogram.
            laser_pulse_fwhm_ps: Laser pulse width (Full Width Half Max).
        """
        # Hardware Parameters
        self.bin_res = bin_resolution_ps * 1e-12  # Convert to seconds
        self.num_bins = int(lidar_period_bins)
        self.period_duration = self.num_bins * self.bin_res

        # Optical Parameters (Based on PDF Table 1 & 2 logic)
        self.c_vacuum = 3e8
        self.refractive_index_water = 1.33
        self.c_water = self.c_vacuum / self.refractive_index_water

        # Laser Pulse Shape
        # Sigma derived from FWHM: sigma = FWHM / (2 * sqrt(2 * ln(2)))
        self.sigma_pulse = (laser_pulse_fwhm_ps * 1e-12) / (2 * np.sqrt(2 * np.log(2)))

        # Pre-compute time axis for efficiency
        self.time_axis = np.linspace(0, self.period_duration, self.num_bins)
        # Distance axis (for reference)
        self.dist_axis = 0.5 * self.c_water * self.time_axis

    def compute_bin_probabilities(self, mu_t, sigma_total, signal_photons):
        """
        Implements Eq. 7 from the PDF:
        Integrates the Gaussian PDF over the bin duration to get exact bin probability.

        Args:
            mu_t: Time of flight (center of the peak) in seconds.
            sigma_total: Total system jitter + pulse width (sigma).
            signal_photons: Average number of photons expected in this pulse (P_pp).
        """
        # Bin edges
        t_start = self.time_axis
        t_end = self.time_axis + self.bin_res

        # CDF calculation using Error Function (Eq 7 logic)
        # P(bin) = P_pp/2 * [erf((t_end - mu)/sig*sqrt(2)) - erf((t_start - mu)/sig*sqrt(2))]
        denom = sigma_total * np.sqrt(2)
        prob_dist = 0.5 * (erf((t_end - mu_t) / denom) - erf((t_start - mu_t) / denom))

        return prob_dist * signal_photons

    def generate_sample(
        self, target_bin_index=None, turbidity_level="medium", signal_strength="random"
    ):
        """
        Generates a single training pair: (Noisy Histogram, Ground Truth Gaussian).

        Args:
            target_bin_index: If None, picks random depth.
            turbidity_level: 'low', 'medium', 'high' (Controls backscatter/attenuation).
            signal_strength: 'random' or specific photon count float.

        Returns:
            noisy_histogram: The input for the U-Net (1D array).
            clean_target: The Ground Truth Gaussian PDF (1D array).
            metadata: Dict containing SNR, depth, etc.
        """

        # 1. Define Environmental Parameters based on Turbidity
        if turbidity_level == "low":  # Clear water
            attenuation_c = 1 / 20  # 1/m
            backscatter_amp = 5
            backscatter_decay = 2.0
        elif turbidity_level == "medium":  # Coastal water
            attenuation_c = 1 / 5
            backscatter_amp = 10
            backscatter_decay = 4.0
        else:  # Turbid/Harbor
            attenuation_c = 1 / 0.5
            backscatter_amp = 50
            backscatter_decay = 6.0

        # 2. Determine Target Depth (Ground Truth)
        if target_bin_index is None:
            target_bin_index = np.random.randint(10, self.num_bins / 8)

        target_time = self.time_axis[target_bin_index]
        target_dist = self.dist_axis[target_bin_index]

        # 3. Calculate Signal Return Strength (P_pp in Eq 1)
        # Apply Beer-Lambert Law: exp(-2 * c * R)
        # And inverse square law: 1 / R^2 (clamped to avoid div by zero)
        geometric_loss = 1.0 / (max(target_dist, 0.5) ** 2)
        transmission_loss = np.exp(-2 * attenuation_c * target_dist)

        if signal_strength == "random":
            base_reflectivity = np.random.uniform(50, 200)
        else:
            base_reflectivity = signal_strength

        avg_signal_photons = base_reflectivity * geometric_loss * transmission_loss

        # 4. Generate Signal Component (The Target Peak)
        # PDF Eq 7: Gaussian spread over bins
        signal_profile = self.compute_bin_probabilities(
            target_time, self.sigma_pulse, avg_signal_photons
        )

        # 5. Generate Backscatter Component (The "Fog")
        # Model: A_bsc * exp(-k * t) / (t^2 offset) - simplified to exponential for 1D
        # Note: In real SPADs, backscatter saturates early bins.
        backscatter_profile = backscatter_amp * np.exp(
            -backscatter_decay * self.time_axis * 1e8
        )  # scaling time for decay

        # 6. Generate Ambient Noise (DC + Background - Eq 3)
        ambient_level = np.random.uniform(0.1, 10)  # Random solar/dark noise floor
        noise_floor = np.full(self.num_bins, ambient_level)

        # 7. Combine to get Rate Function (Lambda)
        # lambda(t) = Signal(t) + Backscatter(t) + Noise
        total_rate_function = signal_profile + backscatter_profile + noise_floor

        # 8. Simulate Photon Detection (Poisson Process)
        # In a real SPAD, we accumulate over N frames.
        # We simulate N frames simply by treating the rate as the mean of a Poisson distribution.
        # If N_frames is high, Poisson is accurate.
        noisy_histogram = np.random.poisson(total_rate_function)

        # 9. Generate Ground Truth Label (Soft Gaussian)
        # For the U-Net, we want a clean probability distribution, not a scalar distance.
        # We generate a normalized Gaussian centered at the true bin.
        # We use a fixed "ideal" sigma for the label (e.g., 2-3 bins wide) to help the network learn.
        label_sigma = 3 * self.bin_res
        ground_truth_profile = self.compute_bin_probabilities(
            target_time, label_sigma, 1.0
        )

        # Normalize Ground Truth to sum to 1 (probability distribution)
        if np.sum(ground_truth_profile) > 0:
            ground_truth_profile /= np.sum(ground_truth_profile)

        # Normalize Input Histogram (Standard practice for Neural Networks)
        # We usually max-normalize or standardize. Max-norm is safer for histograms.
        max_val = np.max(noisy_histogram)
        if max_val > 0:
            norm_input = noisy_histogram / max_val
        else:
            norm_input = noisy_histogram

        metadata = {
            "target_dist_m": target_dist,
            "signal_photons": avg_signal_photons,
            "total_counts": np.sum(noisy_histogram),
            "turbidity": turbidity_level,
        }

        return norm_input, ground_truth_profile, metadata


# ==========================================
# PART 2: The 1D U-Net Architecture
# ==========================================
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling (Encoder)
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))

        # Upsampling (Decoder)
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # 256 because 128(up) + 128(skip)

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(32, 16)

        # Output Layer
        self.outc = nn.Conv1d(16, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4) # Bottleneck

        # Decoder with Skip Connections
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1) # Skip connection
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return self.sigmoid(logits)

# ==========================================
# PART 3: Data Generation & Training Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--checkpoint', type=str, default='unet_checkpoint.pth', help='Path to checkpoint file')
    args = parser.parse_args()

    # 1. Generate Dataset
    print("Generating Simulation Dataset...")
    sim = UnderwaterSPADSimulator(bin_resolution_ps=104.17, lidar_period_bins=1600)
    X_data = []
    Y_data = []
    DATASET_SIZE = 50000

    for i in range(DATASET_SIZE):
        # Mix turbidity levels evenly
        turbidity = np.random.choice(['low', 'low', 'medium', 'high'])
        x, y, _ = sim.generate_sample(turbidity_level=turbidity)
        X_data.append(x)
        Y_data.append(y)

    # Convert to Tensors [Batch, Channels, Length]
    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(1) # Add channel dim -> [N, 1, 1600]
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32).unsqueeze(1) # Add channel dim -> [N, 1, 1600]

    # Create DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * DATASET_SIZE)
    val_size = DATASET_SIZE - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Setup Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = UNet1D().to(device)
    criterion = nn.MSELoss() # MSE is excellent for regression of the Gaussian shape
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    train_losses = []

    if args.eval:
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            exit(1)
    else:
        num_epochs = 100
        print("Starting Training...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")

    # ==========================================
    # PART 4: Evaluation & Visualization
    # ==========================================
    model.eval()
    print("\nEvaluating on Validation Set...")

    # Comprehensive evaluation metrics
    val_losses = []
    distance_errors = []
    snr_improvements = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_dev = inputs.to(device)
            targets_dev = targets.to(device)

            # Forward pass
            outputs = model(inputs_dev)
            val_loss = criterion(outputs, targets_dev)
            val_losses.append(val_loss.item())

            # Convert to numpy for metric calculation
            preds_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            # Calculate distance error for each sample in batch
            for i in range(len(preds_np)):
                pred_curve = preds_np[i, 0]
                true_curve = targets_np[i, 0]

                # Calculate centroids
                bins = np.arange(1600)

                # Predicted peak
                mask_pred = pred_curve > (0.1 * pred_curve.max())
                if np.sum(pred_curve[mask_pred]) > 0:
                    pred_centroid = np.sum(bins[mask_pred] * pred_curve[mask_pred]) / np.sum(pred_curve[mask_pred])
                else:
                    pred_centroid = np.argmax(pred_curve)

                # True peak
                true_peak = np.argmax(true_curve)

                # Distance error in bins
                error = abs(pred_centroid - true_peak)
                distance_errors.append(error)

                # SNR improvement calculation
                input_noise = np.std(inputs_np[i, 0, :100])  # Early bins (noise region)
                input_signal = np.max(inputs_np[i, 0])
                output_signal = np.max(pred_curve)
                output_noise = np.std(pred_curve[:100])

                if input_noise > 0 and output_noise > 0:
                    input_snr = input_signal / input_noise
                    output_snr = output_signal / output_noise
                    snr_improvement = output_snr / input_snr if input_snr > 0 else 1.0
                    snr_improvements.append(snr_improvement)

    # Calculate statistics
    avg_val_loss = np.mean(val_losses)
    median_error = np.median(distance_errors)
    mean_error = np.mean(distance_errors)
    std_error = np.std(distance_errors)
    mae = np.mean(distance_errors)
    rmse = np.sqrt(np.mean(np.array(distance_errors)**2))
    avg_snr_improvement = np.mean(snr_improvements)

    # Convert bin errors to distance (meters)
    bin_to_meters = 0.5 * (3e8 / 1.33) * (104.17e-12)
    mean_error_m = mean_error * bin_to_meters
    median_error_m = median_error * bin_to_meters
    rmse_m = rmse * bin_to_meters

    print(f"\n{'='*50}")
    print("VALIDATION METRICS")
    print(f"{'='*50}")
    print(f"Average Validation Loss: {avg_val_loss:.6f}")
    print("\nDistance Accuracy:")
    print(f"  Mean Error: {mean_error:.2f} bins ({mean_error_m*1000:.2f} mm)")
    print(f"  Median Error: {median_error:.2f} bins ({median_error_m*1000:.2f} mm)")
    print(f"  Std Error: {std_error:.2f} bins")
    print(f"  MAE: {mae:.2f} bins")
    print(f"  RMSE: {rmse:.2f} bins ({rmse_m*1000:.2f} mm)")
    print("\nSignal Quality:")
    print(f"  Avg SNR Improvement: {avg_snr_improvement:.2f}x")
    print(f"{'='*50}\n")

    # Pick a specific challenging sample from validation
    test_inputs, test_targets = next(iter(val_loader))
    test_inputs = test_inputs.to(device)
    with torch.no_grad():
        predictions = model(test_inputs).cpu().numpy()

    # 1. Plot 5 Samples
    fig_samples = plt.figure(figsize=(16, 20))
    for i in range(5):
        if i >= len(test_inputs):
            break

        # Input
        plt.subplot(5, 2, 2*i + 1)
        plt.bar(range(1600), test_inputs[i, 0].cpu().numpy(), color='gray', width=1.0, label='Raw SPAD Input')
        plt.title(f"Sample {i+1}: Input")
        plt.ylabel("Norm. Counts")
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # Output
        plt.subplot(5, 2, 2*i + 2)
        plt.plot(test_targets[i, 0].cpu().numpy(), color='green', linestyle='--', linewidth=2, label='Ground Truth')
        plt.plot(predictions[i, 0], color='red', linewidth=2, label='Prediction')
        plt.title(f"Sample {i+1}: Output")
        plt.ylabel("Probability")
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("unet_samples.png", dpi=150)
    plt.close()

    # 2. Plot Metrics
    fig_metrics = plt.figure(figsize=(16, 12))

    # Subplot 1: Error distribution
    plt.subplot(2, 2, 1)
    plt.hist(distance_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(median_error, color='red', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}')
    plt.axvline(mean_error, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
    plt.title("Distance Error Distribution")
    plt.xlabel("Error (bins)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    # Subplot 2: Training Convergence (only if not eval mode or if we have history)
    if not args.eval and len(train_losses) > 0:
        plt.subplot(2, 2, 2)
        plt.plot(train_losses[1:], marker='o', color='blue', label='Training Loss')
        plt.title("Training Loss Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(alpha=0.3)

    # Subplot 3: SNR Improvement Distribution
    plt.subplot(2, 2, 3)
    plt.hist(snr_improvements, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(avg_snr_improvement, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {avg_snr_improvement:.2f}x')
    plt.title("SNR Improvement Distribution")
    plt.xlabel("SNR Improvement Factor")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    # Subplot 4: Cumulative Error Distribution
    plt.subplot(2, 2, 4)
    sorted_errors = np.sort(distance_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative, linewidth=2, color='purple')
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(90, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(95, color='gray', linestyle='--', alpha=0.5)
    plt.title("Cumulative Error Distribution")
    plt.xlabel("Error (bins)")
    plt.ylabel("Cumulative Percentage (%)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("unet_metrics.png", dpi=150)
    plt.close()

    print("Evaluation complete! Results saved to 'unet_samples.png' and 'unet_metrics.png'")

    # Calculate Centroid (Center of Mass) for sample predictions
    print(f"\n{'='*50}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*50}")
    for idx in range(min(5, len(predictions))):
        pred_curve = predictions[idx, 0]
        bins = np.arange(1600)
        # Simple threshold to remove noise floor from centroid calc
        mask = pred_curve > (0.1 * pred_curve.max())
        if np.sum(pred_curve[mask]) > 0:
            centroid = np.sum(bins[mask] * pred_curve[mask]) / np.sum(pred_curve[mask])
        else:
            centroid = np.argmax(pred_curve)

        # Calculate True Peak
        true_curve = test_targets[idx, 0].cpu().numpy()
        true_peak = np.argmax(true_curve)
        error = abs(centroid - true_peak)
        error_m = error * bin_to_meters

        print(f"Sample {idx+1}:")
        print(f"  True Peak: {true_peak} bins")
        print(f"  Predicted: {centroid:.2f} bins")
        print(f"  Error: {error:.2f} bins ({error_m*1000:.2f} mm)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
