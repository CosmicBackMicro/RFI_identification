import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def find_median_std(data):
    """
    Replicates the logic in findStats.c:findMedianStd
    Calculates median first, then calculates standard deviation relative to that median.
    """
    if len(data) == 0:
        return 0.0, 0.0
    median = np.median(data)
    # The C code uses: sum_sq += (data[i] - median) * (data[i] - median)
    std = np.sqrt(np.mean((data - median)**2))
    return median, std

def in_chan_outlier_iter_demo(data, n_sigma=3.0, max_iterations=15):
    """
    Replicates the iterative outlier detection logic from identification.c:inChanOutlierIter
    Returns snapshots of each iteration for visualization.
    """
    nsamp = len(data)
    mask = np.zeros(nsamp, dtype=bool)
    snapshots = []
    
    # Constants from C code
    STD_CHANGE_THRESHOLD = 0.0001
    MEDIAN_CHANGE_THRESHOLD = 1e-6
    EPS_STD = 1e-12
    
    valid_indices = np.where(~mask)[0]
    last_median, last_std = 0.0, 0.0
    
    for iter_idx in range(max_iterations):
        if len(valid_indices) < 3:
            break
            
        current_median, current_std = find_median_std(data[valid_indices])
        
        # Save snapshot of the state BEFORE flagging in this iteration
        snapshots.append({
            'iteration': iter_idx,
            'median': current_median,
            'std': current_std,
            'mask': mask.copy(),
            'valid_data': data[valid_indices].copy(),
            'full_data': data.copy()
        })
        
        if current_std <= EPS_STD:
            break
            
        # Convergence check (from 2nd iteration onwards)
        if iter_idx > 0:
            median_change = abs(current_median - last_median)
            denom = last_std if last_std > EPS_STD else EPS_STD
            std_change_rate = abs(current_std - last_std) / denom
            
            if median_change < MEDIAN_CHANGE_THRESHOLD and std_change_rate < STD_CHANGE_THRESHOLD:
                # Still record the final convergence but the loop will stop
                break
        
        upper_bound = current_median + n_sigma * current_std
        lower_bound = current_median - n_sigma * current_std
        
        new_outliers = 0
        for i in valid_indices:
            if data[i] > upper_bound or data[i] < lower_bound:
                mask[i] = True
                new_outliers += 1
        
        if new_outliers == 0:
            break
            
        # Update valid indices for next iteration
        valid_indices = np.where(~mask)[0]
        last_median, last_std = current_median, current_std
        
    return snapshots

def visualize_iteration_steps(snapshots, n_sigma, output_path="gaussian_iteration_demo.png"):
    # Get initial and final state
    snap_init = snapshots[0]
    snap_final = snapshots[-1]
    data = snap_init['full_data']
    mask = snap_final['mask']
    valid_data = snap_final['valid_data']
    num_iters = len(snapshots)
    
    # Final stats
    mu = snap_final['median']
    sigma = snap_final['std']
    
    # Style settings
    TITLE_FONT = 38
    LABEL_FONT = 38
    TICK_FONT = 38
    LEGEND_FONT = 38

    fig, ax = plt.subplots(figsize=(22, 14))
    
    # 1. Initial Histogram (Big Red)
    counts_init, bins, _ = ax.hist(data, bins=100, color='red', alpha=0.75, 
                                   label='Initial Distribution (Before Iteration)')
    bin_width = bins[1] - bins[0]
    
    # 2. Flagged Data Histogram (Green dashed outline)
    flagged_count = np.sum(mask)
    if flagged_count > 0:
        ax.hist(data[mask], bins=bins.tolist(), histtype='step', linestyle='--', color='green', 
                linewidth=5, label=f'Removed RFI ({flagged_count} pixels, {flagged_count/len(data)*100:.1f}%)')
    
    # 3. Fit Gaussian Curve
    x = np.linspace(np.min(data), np.max(data), 1000)
    y = norm.pdf(x, mu, sigma) * len(valid_data) * bin_width
    ax.plot(x, y, color='black', linewidth=6, label=f'Fitted Gaussian (σ={sigma:.4f}, Iters={num_iters})')
    
    # 4. Vertical line for Mean (mu)
    ax.axvline(mu, color='black', linestyle='-', linewidth=4, label=f'Fitted Mean (μ={mu:.4f})')
    
    # 5. Iteration Thresholds (Blue dash-dot)
    upper = mu + n_sigma * sigma
    lower = mu - n_sigma * sigma
    ax.axvline(upper, color='blue', linestyle='-.', linewidth=4, label=f'Final {n_sigma}σ Threshold')
    ax.axvline(lower, color='blue', linestyle='-.', linewidth=4)
    
    # Aesthetic adjustments
    ax.set_title(rf"Gaussian Sigma Clipping ({n_sigma}$\sigma$)", fontsize=TITLE_FONT, pad=25)
    ax.set_xlabel("Pixel Intensity", fontsize=LABEL_FONT)
    ax.set_ylabel("Pixel Counts", fontsize=LABEL_FONT)
    ax.tick_params(labelsize=TICK_FONT)
    
    # Consolidation: Adjust legend location and size
    ax.legend(fontsize=LEGEND_FONT, loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"Professional single-panel visualization saved to {output_path}")

if __name__ == "__main__":
    # 1. Generate synthetic data: Noise + RFI
    np.random.seed(42)
    nsamp = 2048
    noise_sigma = 1.0
    noise_median = 10.0
    data = np.random.normal(noise_median, noise_sigma, nsamp)
    
    # Add some strong outliers (RFI spikes)
    num_outliers = 50
    outlier_indices = np.random.choice(nsamp, num_outliers, replace=False)
    data[outlier_indices] += np.random.uniform(5, 15, num_outliers)
    
    # Add some weaker outliers to see multi-step iteration
    num_weak = 30
    weak_indices = np.random.choice(nsamp, num_weak, replace=False)
    data[weak_indices] += np.random.uniform(3.5, 4.5, num_weak)

    # 2. Run the iterative process
    n_sigma = 3.0
    snapshots = in_chan_outlier_iter_demo(data, n_sigma=n_sigma)
    
    # 3. Print summary
    print(f"Total iterations: {len(snapshots)}")
    for snap in snapshots:
        print(f"Iter {snap['iteration']}: median={snap['median']:.4f}, std={snap['std']:.4f}, flagged_total={np.sum(snap['mask'])}")
    
    # 4. Visualize
    visualize_iteration_steps(snapshots, n_sigma)
