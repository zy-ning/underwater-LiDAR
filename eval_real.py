import argparse
import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import MobileNet1D, Performer1D, ResNet1D, Transformer1D, UNet1D, mc_predict


def parse_log_file(log_file, period_bins=1600):
    """
    Parse a log file and extract photon counts.

    Returns both per-period histograms and their average.
    """
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

    # Calculate number of complete periods
    num_periods = len(photon_counts) // period_bins

    if num_periods == 0:
        # Not enough data for one full period, pad what we have
        if len(photon_counts) < period_bins:
            photon_counts = np.pad(photon_counts, (0, period_bins - len(photon_counts)))
        single = photon_counts[:period_bins]
        return np.expand_dims(single, 0), single

    # Reshape into periods and average across them (use only complete periods)
    photon_counts = photon_counts[: num_periods * period_bins]
    photon_counts = photon_counts.reshape(num_periods, period_bins)
    averaged_counts = np.mean(photon_counts, axis=0)

    return photon_counts, averaged_counts


def preprocess_real_data(photon_counts):
    """
    Normalize real data for U-Net input.

    Args:
        photon_counts: Raw photon counts array

    Returns:
        Normalized tensor ready for model input [1, 1, 1600]
    """
    # Max normalization (same as training)
    max_val = np.max(photon_counts)
    if max_val > 0:
        normalized = photon_counts / max_val
    else:
        normalized = photon_counts

    # Convert to tensor with batch and channel dimensions
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


def build_model(model_name, dropout=0.1):
    """Create an inference model instance based on a string name."""
    if model_name == "unet":
        return UNet1D(p_drop=0.0)
    if model_name == "mcd_unet":
        return UNet1D(p_drop=dropout)
    if model_name == "mobilenet":
        return MobileNet1D(L=1600)
    if model_name == "resnet":
        return ResNet1D()
    if model_name == "transformer":
        return Transformer1D(L=1600)
    if model_name == "performer":
        return Performer1D(L=1600)
    raise ValueError(f"Unknown model name: {model_name}")


def run_recursive_kalman(
    predictions,
    variances,
    process_noise_q=1e-5,
    uncertainty_power=2.0,
    drop_threshold=None,
    drop_by_mean=True,
    drop_factor=1.5,
    outlier_alpha=2.0,
    outlier_drop=None,
    center_method="median",
):
    """
    Performs Heteroscedastic Recursive Kalman Filtering on the waveform sequences.

    Mathematically:
    1. Prediction Step:
       x_{t|t-1} = x_{t-1|t-1} (Constant position model)
       P_{t|t-1} = P_{t-1|t-1} + Q

    2. Update Step:
       K_t = P_{t|t-1} / (P_{t|t-1} + R_t)
       x_{t|t} = x_{t|t-1} + K_t * (z_t - x_{t|t-1})
       P_{t|t} = (1 - K_t) * P_{t|t-1}

    Args:
        predictions (np.array): [N_periods, L_bins] - The measurements (z)
        variances (np.array):   [N_periods, L_bins] - The measurement noise (R) derived from MCD
        process_noise_q (float): Hyperparameter trusting the evolution vs measurement.
        uncertainty_power (float): Scales impact of large variance; >1 down-weights uncertain frames.
        drop_threshold (float|None): If set, drop periods whose mean variance exceeds this value.
        drop_by_mean (bool): If True, drop periods whose mean variance exceeds (drop_factor * global mean).
        drop_factor (float): Multiplier for global mean variance when drop_by_mean is enabled.
        outlier_alpha (float): Scales variance by distance from cluster center of peak bins; higher penalizes outliers.
        outlier_drop (float|None): If set, drop periods whose peak-bin distance from center exceeds this many bins.
        center_method (str): "median" or "mean" for cluster center of peak bins.

    Returns:
        final_estimate (np.array): [L_bins]
        final_covariance (np.array): [L_bins]
    """
    n_periods, n_bins = predictions.shape

    # variances: [N, L]
    R = variances**uncertainty_power  # >1.0 makes high-variance frames count less

    # Penalize outliers in peak location relative to cluster center
    peak_bins = np.argmax(predictions, axis=1)
    if center_method == "mean":
        center_bin = float(np.mean(peak_bins))
    else:
        center_bin = float(np.median(peak_bins))
    dist = np.abs(peak_bins - center_bin)

    if outlier_alpha is not None and outlier_alpha > 0:
        scale = 1.0 + outlier_alpha * (dist / max(1.0, n_bins))
        R = R * scale[:, None]

    if outlier_drop is not None:
        outlier_mask = dist <= outlier_drop
        predictions = predictions[outlier_mask]
        R = R[outlier_mask]
        if len(predictions) == 0:
            raise ValueError("All periods dropped by outlier threshold")

    # Drop periods that are too uncertain
    if drop_by_mean or drop_threshold is not None:
        per_period_var = R.mean(axis=1)
        global_mean_var = float(per_period_var.mean())
        mask = np.ones_like(per_period_var, dtype=bool)
        if drop_by_mean:
            mask &= per_period_var <= drop_factor * global_mean_var
        if drop_threshold is not None:
            mask &= per_period_var <= drop_threshold
        predictions = predictions[mask]
        R = R[mask]
        if len(predictions) == 0:
            raise ValueError("All periods dropped by uncertainty threshold")

    # --- Initialization (t=0) ---
    # We initialize the state x with the first measurement
    # We initialize the error covariance P with the first measurement variance
    x_est = predictions[0].copy()
    P_est = R[0].copy()

    # Iterate through remaining periods (t=1 to N)
    for t in range(1, len(predictions)):
        # --- 1. Prediction Step ---
        # Assume static target: x_pred = x_prev
        # Increase uncertainty by process noise Q
        P_pred = P_est + process_noise_q

        # --- 2. Measurement Data for t ---
        z_t = predictions[t]  # Measurement
        R_t = R[t]  # Measurement Noise (Aleatoric + Epistemic from MCD)

        # --- 3. Update Step (Kalman Gain) ---
        # K = P_pred / (P_pred + R)
        # Add epsilon to prevent division by zero
        K_t = P_pred / (P_pred + R_t + 1e-12)

        # Update State Estimate
        x_est = x_est + K_t * (z_t - x_est)

        # Update Error Covariance
        P_est = (1.0 - K_t) * P_pred

    return x_est, P_est


def extract_distance_from_prediction(
    prediction,
    bin_resolution_ps=104.17,
    speed_of_light=299792458 / 1.33,
    distance_bias=1.52,
    top_p=0.1,
    sweet_zone=(100, 300),
    forced_peak_bin=None,
):
    """
    Extract distance estimate from U-Net prediction using top-p filtering.

    Args:
        prediction: Model output [1600] array
        bin_resolution_ps: Time resolution per bin in picoseconds
        speed_of_light: Speed of light in the medium (m/s)
        distance_bias: Distance bias correction in meters
        top_p: Cumulative probability threshold for filtering (default 0.25)

    Returns:
        dict with distance estimates and metrics
    """
    # bins = np.arange(len(prediction))

    if forced_peak_bin is None:
        # Top-p filtering: Find the first peak in the top-p probability mass
        sorted_indices = np.argsort(prediction)[::-1]
        sorted_probs = prediction[sorted_indices]

        # Normalize to probability distribution
        prob_sum = np.sum(sorted_probs)
        if prob_sum > 0:
            sorted_probs_norm = sorted_probs / prob_sum
        else:
            sorted_probs_norm = sorted_probs

        # Calculate cumulative probability
        cumsum_probs = np.cumsum(sorted_probs_norm)

        # Find bins that are in top-p
        top_p_mask = cumsum_probs <= top_p
        top_p_bins = sorted_indices[top_p_mask]

        # Prefer earliest candidate inside sweet_zone; otherwise earliest overall; fallback to argmax
        if len(top_p_bins) > 0:
            if sweet_zone is not None and len(sweet_zone) == 2:
                zone_start, zone_end = sweet_zone
                in_zone = [b for b in top_p_bins if zone_start <= b <= zone_end]
                if len(in_zone) > 0:
                    peak_bin = np.min(in_zone)
                else:
                    peak_bin = np.min(top_p_bins)
            else:
                peak_bin = np.min(top_p_bins)
        else:
            # Fallback: use global maximum
            peak_bin = np.argmax(prediction)
    else:
        peak_bin = int(forced_peak_bin)
        top_p_bins = np.array([peak_bin])

    # Convert bins to distance
    bin_resolution_s = bin_resolution_ps * 1e-12

    # Distance = 0.5 * c * t (round trip) with bias correction
    distance_m = 0.5 * speed_of_light * bin_resolution_s * peak_bin - distance_bias

    # Calculate confidence metrics
    peak_value = prediction[peak_bin]
    background_noise = np.mean(prediction[:100])  # First 100 bins as noise estimate
    snr = peak_value / background_noise if background_noise > 0 else float("inf")

    # Calculate how many candidate peaks were in top-p
    num_candidates = len(top_p_bins)

    return {
        "peak_bin": peak_bin,
        "distance_m": distance_m,
        "peak_value": peak_value,
        "snr": snr,
        "num_candidates": num_candidates,
        "top_p_bins": sorted(top_p_bins.tolist()) if len(top_p_bins) > 0 else [],
    }


def evaluate_single_file(
    model, log_file, device, aggregation="pre_avg", model_name="unet", mc_samples=30
):
    """
    Evaluate model on a single log file.

    Args:
        model: Trained U-Net model
        log_file: Path to log file
        device: torch device

    Returns:
        tuple of (input_data, prediction, metrics, filename)
    """
    # Determine medium from file path and set parameters
    if "air" in log_file:
        distance_bias = 2.05
        speed_of_light = 299792458  # meters per second
    elif "water" in log_file:
        distance_bias = 1.52
        speed_of_light = 299792458 / 1.33  # meters per second
    else:
        # Default to water
        distance_bias = 1.52
        speed_of_light = 299792458 / 1.33

    # Parse log file
    periods, averaged_counts = parse_log_file(log_file)

    # Choose aggregation strategy
    if aggregation == "pre_avg":
        input_tensor = preprocess_real_data(averaged_counts).to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0, 0]
        metrics = extract_distance_from_prediction(
            prediction, speed_of_light=speed_of_light, distance_bias=distance_bias
        )
        photon_counts = averaged_counts

    else:
        # If no multiple periods are available, fall back to pre_avg behavior
        if periods.shape[0] <= 1:
            input_tensor = preprocess_real_data(averaged_counts).to(device)
            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor).cpu().numpy()[0, 0]
            metrics = extract_distance_from_prediction(
                prediction, speed_of_light=speed_of_light, distance_bias=distance_bias
            )
            photon_counts = averaged_counts
        else:
            preds = []
            vars_ = []
            for period_hist in periods:
                input_tensor = preprocess_real_data(period_hist).to(device)
                if aggregation == "kalman":
                    mean_pred, var_pred = mc_predict(model, input_tensor, T=mc_samples)
                    preds.append(mean_pred.cpu().numpy()[0, 0])
                    vars_.append(var_pred.cpu().numpy()[0, 0])
                else:
                    model.eval()
                    with torch.no_grad():
                        pred = model(input_tensor).cpu().numpy()[0, 0]
                    preds.append(pred)

            preds = np.stack(preds)

            if aggregation == "post_avg":
                prediction = np.mean(preds, axis=0)
                metrics = extract_distance_from_prediction(
                    prediction,
                    speed_of_light=speed_of_light,
                    distance_bias=distance_bias,
                )

            elif aggregation == "maj_vote":
                argmax_bins = [int(np.argmax(p)) for p in preds]
                counts = Counter(argmax_bins)
                max_count = max(counts.values())
                majority_candidates = [b for b, c in counts.items() if c == max_count]
                majority_bin = min(majority_candidates)
                prediction = np.mean(preds, axis=0)
                metrics = extract_distance_from_prediction(
                    prediction,
                    speed_of_light=speed_of_light,
                    distance_bias=distance_bias,
                    forced_peak_bin=majority_bin,
                )
                metrics["majority_bin"] = majority_bin
                metrics["majority_vote_counts"] = dict(counts)

            elif aggregation == "kalman":
                if model_name != "mcd_unet":
                    # Fallback to post_avg if user requested kalman without MC-Dropout model
                    prediction = np.mean(preds, axis=0)
                    metrics = extract_distance_from_prediction(
                        prediction,
                        speed_of_light=speed_of_light,
                        distance_bias=distance_bias,
                    )
                else:
                    vars_ = np.stack(vars_)
                    prediction, fused_var = run_recursive_kalman(preds, vars_)
                    metrics = extract_distance_from_prediction(
                        prediction,
                        speed_of_light=speed_of_light,
                        distance_bias=distance_bias,
                    )
                    metrics["fused_variance_mean"] = float(np.mean(fused_var))
                    metrics["fused_variance_max"] = float(np.max(fused_var))
                    metrics["fused_variance_min"] = float(np.min(fused_var))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            photon_counts = averaged_counts

    # Add filename info
    filename = os.path.basename(log_file)

    return photon_counts, prediction, metrics, filename


def plot_result(photon_counts, prediction, metrics, filename, save_path=None):
    """
    Plot input and output for a single file.

    Args:
        photon_counts: Raw input data
        prediction: Model prediction
        metrics: Distance metrics dict
        filename: Original filename
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    bins = np.arange(len(photon_counts))

    # Plot 1: Raw input
    ax1.bar(
        bins, photon_counts, color="gray", width=1.0, alpha=0.7, label="Raw SPAD Data"
    )
    ax1.set_xlabel("Bin Number")
    ax1.set_ylabel("Photon Counts")
    ax1.set_title(f"Input: {filename}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Model prediction
    ax2.plot(bins, prediction, color="red", linewidth=2, label="U-Net Prediction")
    ax2.axvline(
        metrics["peak_bin"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Selected Peak: bin {metrics['peak_bin']}",
    )

    # Mark all top-p candidate peaks
    if "top_p_bins" in metrics and len(metrics["top_p_bins"]) > 0:
        for candidate_bin in metrics["top_p_bins"]:
            if candidate_bin != metrics["peak_bin"]:
                ax2.axvline(
                    candidate_bin,
                    color="lightblue",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.5,
                )

    ax2.set_xlabel("Bin Number")
    ax2.set_ylabel("Probability")
    ax2.set_title("U-Net Output - Distance Estimate (Top-P Filtering)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add text box with metrics
    textstr = "\n".join(
        [
            # f"Distance: {metrics['distance_m']*100:.2f} cm",
            f"Peak Bin: {metrics['peak_bin']}",
            f"SNR: {metrics['snr']:.2f}",
            f"Peak Value: {metrics['peak_value']:.4f}",
            f"Candidates: {metrics['num_candidates']}",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax2.text(
        0.98,
        0.97,
        textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Model on real SPAD data"
    )
    # parser.add_argument('--checkpoint', type=str, default='unet_checkpoint.pth',
    #                     help='Path to model checkpoint')
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="checkpoint_{model}.pth",
        help="Template for per-model checkpoints (use {model} placeholder)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="unet",
        help="Comma-separated models to evaluate: unet,mcd_unet,resnet,mobilenet,transformer,performer",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout prob for mcd_unet"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="MC samples for mcd_unet kalman fusion",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="pre_avg",
        choices=["pre_avg", "post_avg", "maj_vote", "kalman"],
        help="Histogram aggregation across periods",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to specific log file (if not provided, searches data/ directory)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="data", help="Directory to search for log files"
    )
    parser.add_argument(
        "--medium",
        type=str,
        default="water",
        choices=["water", "air"],
        help="Medium type (water or air) for refractive index",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots instead of saving"
    )

    args = parser.parse_args()

    # Normalize model list once for reuse
    args.models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.models_list:
        print("Error: No models specified for evaluation.")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # Set refractive index
    # refractive_index = 1.33 if args.medium == 'water' else 1.0
    # print(f"Medium: {args.medium} (n={refractive_index})")

    # Find log files
    if args.log_file:
        log_files = [args.log_file]
    else:
        log_files = glob.glob(os.path.join(args.log_dir, "**", "*.log"), recursive=True)
        if not log_files:
            print(f"No .log files found in {args.log_dir}")
            return
        print(f"Found {len(log_files)} log files")

    # Evaluate each requested model independently
    for model_name in args.models_list:
        print(f"\n{'=' * 70}")
        print(f"EVALUATION RESULTS - {model_name}")
        print(f"{'=' * 70}")

        checkpoint_path = args.checkpoint_template.format(model=model_name)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint missing for {model_name}: {checkpoint_path}")
            continue

        model = build_model(model_name, dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

        model_output_dir = (
            args.output_dir
            if len(args.models_list) == 1
            else os.path.join(args.output_dir, model_name)
        )
        if not args.show_plots:
            os.makedirs(model_output_dir, exist_ok=True)

        results = []

        for i, log_file in enumerate(log_files):
            print(
                f"\n[{i + 1}/{len(log_files)}] Processing: {os.path.basename(log_file)}"
            )

            try:
                photon_counts, prediction, metrics, filename = evaluate_single_file(
                    model,
                    log_file,
                    device,
                    aggregation=args.aggregation,
                    model_name=model_name,
                    mc_samples=args.mc_samples,
                )

                print(f"  Distance:       {metrics['distance_m'] * 100:6.2f} cm")
                print(f"  Peak Bin:       {metrics['peak_bin']:6d}")
                print(f"  SNR:            {metrics['snr']:6.2f}")
                print(f"  Peak Value:     {metrics['peak_value']:6.4f}")
                print(f"  Candidates:     {metrics['num_candidates']:6d}")

                if args.show_plots:
                    plot_result(photon_counts, prediction, metrics, filename)
                else:
                    save_name = (
                        os.path.splitext(filename)[0] + f"_{model_name}_eval.png"
                    )
                    save_path = os.path.join(model_output_dir, save_name)
                    plot_result(photon_counts, prediction, metrics, filename, save_path)

                results.append(
                    {
                        "model": model_name,
                        "filename": filename,
                        "distance_cm": metrics["distance_m"] * 100,
                        "peak_bin": metrics["peak_bin"],
                        "snr": metrics["snr"],
                        "peak_value": metrics["peak_value"],
                        "num_candidates": metrics["num_candidates"],
                    }
                )

            except Exception as e:
                print(f"  Error processing {log_file}: {str(e)}")
                continue

        if results:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY STATISTICS - {model_name}")
            print(f"{'=' * 70}")
            print(f"Total files processed: {len(results)}")

            distances = [r["distance_cm"] for r in results]
            snrs = [r["snr"] for r in results]

            print("\nDistance Statistics (Top-P First Peak):")
            print(f"  Mean:   {np.mean(distances):6.2f} cm")
            print(f"  Median: {np.median(distances):6.2f} cm")
            print(f"  Std:    {np.std(distances):6.2f} cm")
            print(f"  Min:    {np.min(distances):6.2f} cm")
            print(f"  Max:    {np.max(distances):6.2f} cm")

            print("\nSNR Statistics:")
            print(f"  Mean:   {np.mean(snrs):6.2f}")
            print(f"  Median: {np.median(snrs):6.2f}")
            print(f"  Std:    {np.std(snrs):6.2f}")

            if not args.show_plots:
                import csv

                csv_name = (
                    "evaluation_results.csv"
                    if len(args.models_list) == 1
                    else f"evaluation_results_{model_name}.csv"
                )
                csv_path = os.path.join(model_output_dir, csv_name)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"\nResults saved to {csv_path}")

    print(f"\n{'=' * 70}")
    print("Evaluation complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
