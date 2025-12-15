import numpy as np

def calculate_disorder_percentage(predictions, threshold=0.5):
    """
    Calculate the percentage of disordered residues.
    
    Args:
        predictions: Tensor of shape (L, 2) with [structured_score, disordered_score]
        threshold: Threshold to classify residues as disordered
    Returns:
        Dictionary with different disorder statistics
    """
    # Get disordered scores
    disorder_scores = predictions[:, 1].numpy()
    
    # Count residues with disorder score > threshold
    disordered_count = np.sum(disorder_scores > threshold)
    total_residues = len(disorder_scores)
    disorder_percentage = (disordered_count / total_residues) * 100
    
    return {
        'disorder_percentage': disorder_percentage,
        'disordered_residues': disordered_count,
        'total_residues': total_residues,
        'disorder_scores': disorder_scores
    }