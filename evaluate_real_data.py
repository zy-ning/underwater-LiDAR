import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from u_net import UnderwaterSPADSimulator, UNet1D


def parse_log_file(log_file, period_bins=1600):
    """
    Parse a log file and extract photon counts.

    Args:
        log_file: Path to the log file
        period_bins: Number of bins to read

    Returns:
        numpy array of photon counts
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

    photon_counts = np.array(photon_counts)[:period_bins]

    # Pad if necessary
    if len(photon_counts) < period_bins:
        photon_counts = np.pad(photon_counts, (0, period_bins - len(photon_counts)))

    return photon_counts

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

def extract_distance_from_prediction(prediction, bin_resolution_ps=104.17, refractive_index=1.33, bin_bias=84, top_p=0.25):
    """
    Extract distance estimate from U-Net prediction using top-p filtering.

    Args:
        prediction: Model output [1600] array
        bin_resolution_ps: Time resolution per bin in picoseconds
        refractive_index: Refractive index of medium (1.33 for water, 1.0 for air)
        dist_bias: Distance bias correction in meters
        top_p: Cumulative probability threshold for filtering (default 0.9)

    Returns:
        dict with distance estimates and metrics
    """
    # bins = np.arange(len(prediction))

    # Top-p filtering: Find the first peak in the top-p probability mass
    # Sort bins by prediction value (descending)
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

    # Select the first (earliest in time) bin from top-p candidates
    if len(top_p_bins) > 0:
        peak_bin = np.min(top_p_bins)
    else:
        # Fallback: use global maximum
        peak_bin = np.argmax(prediction)

    # Convert bins to distance
    c_vacuum = 3e8
    c_medium = c_vacuum / refractive_index
    bin_resolution_s = bin_resolution_ps * 1e-12

    # Distance = 0.5 * c * t (round trip) with bias correction
    distance_m = 0.5 * c_medium * bin_resolution_s * (peak_bin - bin_bias)

    # Calculate confidence metrics
    peak_value = prediction[peak_bin]
    background_noise = np.mean(prediction[:100])  # First 100 bins as noise estimate
    snr = peak_value / background_noise if background_noise > 0 else float('inf')

    # Calculate how many candidate peaks were in top-p
    num_candidates = len(top_p_bins)

    return {
        'peak_bin': peak_bin,
        'distance_m': distance_m,
        'peak_value': peak_value,
        'snr': snr,
        'num_candidates': num_candidates,
        'top_p_bins': sorted(top_p_bins.tolist()) if len(top_p_bins) > 0 else []
    }

def evaluate_single_file(model, log_file, device, refractive_index=1.33):
    """
    Evaluate model on a single log file.

    Args:
        model: Trained U-Net model
        log_file: Path to log file
        device: torch device
        refractive_index: Medium refractive index

    Returns:
        tuple of (input_data, prediction, metrics, filename)
    """
    # Parse log file
    photon_counts = parse_log_file(log_file)

    # Preprocess
    input_tensor = preprocess_real_data(photon_counts).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0, 0]

    # Extract distance
    metrics = extract_distance_from_prediction(prediction, refractive_index=refractive_index)

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
    ax1.bar(bins, photon_counts, color='gray', width=1.0, alpha=0.7, label='Raw SPAD Data')
    ax1.set_xlabel('Bin Number')
    ax1.set_ylabel('Photon Counts')
    ax1.set_title(f'Input: {filename}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Model prediction
    ax2.plot(bins, prediction, color='red', linewidth=2, label='U-Net Prediction')
    ax2.axvline(metrics['peak_bin'], color='blue', linestyle='--', linewidth=2, label=f"Selected Peak: bin {metrics['peak_bin']}")

    # Mark all top-p candidate peaks
    if 'top_p_bins' in metrics and len(metrics['top_p_bins']) > 0:
        for candidate_bin in metrics['top_p_bins']:
            if candidate_bin != metrics['peak_bin']:
                ax2.axvline(candidate_bin, color='lightblue', linestyle=':', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Bin Number')
    ax2.set_ylabel('Probability')
    ax2.set_title('U-Net Output - Distance Estimate (Top-P Filtering)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add text box with metrics
    textstr = '\n'.join([
        # f"Distance: {metrics['distance_m']*100:.2f} cm",
        f"Peak Bin: {metrics['peak_bin']}",
        f"SNR: {metrics['snr']:.2f}",
        f"Peak Value: {metrics['peak_value']:.4f}",
        f"Candidates: {metrics['num_candidates']}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained U-Net on real SPAD data')
    parser.add_argument('--checkpoint', type=str, default='unet_checkpoint.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to specific log file (if not provided, searches data/ directory)')
    parser.add_argument('--log-dir', type=str, default='data',
                        help='Directory to search for log files')
    parser.add_argument('--medium', type=str, default='water', choices=['water', 'air'],
                        help='Medium type (water or air) for refractive index')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots instead of saving')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return

    model = UNet1D().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Set refractive index
    refractive_index = 1.33 if args.medium == 'water' else 1.0
    print(f"Medium: {args.medium} (n={refractive_index})")

    # Create output directory
    if not args.show_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    # Find log files
    if args.log_file:
        log_files = [args.log_file]
    else:
        # Search for all .log files in data directory
        log_files = glob.glob(os.path.join(args.log_dir, '**', '*.log'), recursive=True)
        if not log_files:
            print(f"No .log files found in {args.log_dir}")
            return
        print(f"Found {len(log_files)} log files")

    # Evaluate each file
    results = []

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    for i, log_file in enumerate(log_files):
        print(f"\n[{i+1}/{len(log_files)}] Processing: {os.path.basename(log_file)}")

        try:
            photon_counts, prediction, metrics, filename = evaluate_single_file(
                model, log_file, device, refractive_index
            )

            # Print metrics
            print(f"  Distance:       {metrics['distance_m']*100:6.2f} cm")
            print(f"  Peak Bin:       {metrics['peak_bin']:6d}")
            print(f"  SNR:            {metrics['snr']:6.2f}")
            print(f"  Peak Value:     {metrics['peak_value']:6.4f}")
            print(f"  Candidates:     {metrics['num_candidates']:6d}")

            # Save or show plot
            if args.show_plots:
                plot_result(photon_counts, prediction, metrics, filename)
            else:
                save_name = os.path.splitext(filename)[0] + '_eval.png'
                save_path = os.path.join(args.output_dir, save_name)
                plot_result(photon_counts, prediction, metrics, filename, save_path)

            # Store results
            results.append({
                'filename': filename,
                'distance_cm': metrics['distance_m'] * 100,
                'peak_bin': metrics['peak_bin'],
                'snr': metrics['snr'],
                'peak_value': metrics['peak_value'],
                'num_candidates': metrics['num_candidates']
            })

        except Exception as e:
            print(f"  Error processing {log_file}: {str(e)}")
            continue

    # Summary statistics
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Total files processed: {len(results)}")

        distances = [r['distance_cm'] for r in results]
        snrs = [r['snr'] for r in results]

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

        # Save results to CSV
        if not args.show_plots:
            csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to {csv_path}")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
