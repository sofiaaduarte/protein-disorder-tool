import matplotlib.pyplot as plt
from pathlib import Path

def plot_disorder_prediction(centers, predictions, protein_id, threshold=0.5, 
                            output_path=None):
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
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot disorder score
    ax.plot(centers, pred[:, 1], "-", color="red", linewidth=1.5, 
            label='Disordered')
    
    # Add threshold line
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, 
               label=f'Threshold = {threshold}')
    
    # Set limits and labels
    ax.set_xlim([centers.min(), centers.max()])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Amino acid position", fontsize=11)
    ax.set_ylabel("Disorder score", fontsize=11)
    ax.set_title(f"Disorder Prediction - {protein_id}", fontsize=13, 
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    # ax.grid(alpha=0.3)
    
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