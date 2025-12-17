import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_disorder_prediction(centers, predictions, protein_id, threshold=0.5, 
                            output_path=None, highlight_regions=True, style='default',
                            figsize=(12, 5)):
    """
    Plot disorder predictions.
    
    Args:
        centers: Array of center positions
        predictions: Prediction scores (structured, disordered) for each position
        protein_id: Protein identifier (acc)
        threshold: Threshold line used to classify residues as disordered
        smooth_window: Window size for smoothing (0 = no smoothing)
        output_path: Path to save the plot (None = display only)
    """
    pred = predictions.numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Highlight disordered regions with background color
    if highlight_regions:
        disordered_mask = pred[:, 1] > threshold
        # Find continuous regions
        regions = []
        start = None
        for i, is_disordered in enumerate(disordered_mask):
            if is_disordered and start is None:
                start = i
            elif not is_disordered and start is not None:
                regions.append((centers[start], centers[i-1]))
                start = None
        if start is not None:
            regions.append((centers[start], centers[-1]))
        
        # Draw regions
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.2, color='red', zorder=0,
                       edgecolor=None)
    
    # Plot disorder score
    ax.plot(centers, pred[:, 1], "-", color="#d62728", linewidth=2, 
            label='Disorder score', zorder=3)
    
    # Add threshold line
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Threshold = {threshold}', zorder=1)
    
    # Set limits and labels
    ax.set_xlim([centers.min(), centers.max()])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Prediction score")
    ax.set_title(f"Intrinsic Disorder Prediction - {protein_id}")
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    if style != 'minimal':
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()