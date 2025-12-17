"""
Predict disorder from protein embeddings using a trained model
"""
import argparse
import numpy as np
import torch as tr
from pathlib import Path
from src.model import BaseModel
from src.utils import ConfigLoader, predict_sliding_window, get_embedding_size, calculate_disorder_percentage
from src.plms import generate_embedding_from_sequence
from src.plot import plot_disorder_prediction

def parser(): # * ESTA OK!
    parser = argparse.ArgumentParser(
        description='Predict disorder from protein embeddings using a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        # to show the default values in help messages
    )
    parser.add_argument(
        '--fasta', '-f',
        type=str,
        required=True,
        help='Path to FASTA file (will generate embedding on-the-fly)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='ESM2',
        choices=['ESM2', 'ProtT5'], # Later will add ['ProstT5', 'esmc_300m', 'esmc_600m'],
        help='Protein Language Model (pLM) used for generating embeddings. '
             'The disorder prediction model was trained using embeddings from this pLM'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file to save predictions (.csv)'
    )
    parser.add_argument(
        '--plot', '-p',
        type=str,
        default=None,
        help='Output file to save plot (.png or .pdf)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        help='Device to run predictions on (e.g., "cpu", "cuda", "cuda:0", "cuda:1")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def main():
    args = parser()

    # Validate and setup device ------------------------------------------------
    device = args.device.lower()
    
    if device.startswith('cuda') and not tr.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu")
    
    if args.verbose:
        device_name = tr.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'
        print(f"Using device: {device} ({device_name})")

    # Set up model path, config and weights ------------------------------------
    model_dir = Path(f"model/{args.model}/model0/")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    config_path = model_dir / 'config.yaml'
    weights_path = model_dir / 'weights.pk'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load model configuration -------------------------------------------------
    if args.verbose:
        print(f"Model directory: {model_dir}")
        print(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()
    window_len = config.get('win_len', 13)
    threshold = config.get('threshold', 0.5)
    
    # Initialize model ---------------------------------------------------------
    if args.verbose:
        print(f"Loading model from: {weights_path}")
    categories = ['structured', 'disordered']
    model = BaseModel(
        len(categories),
        lr=config['lr'],
        device=device,
        emb_size=get_embedding_size(config.get('plm', 'esm2')),
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['n_resnet']
    )
    model.load_state_dict(tr.load(weights_path, map_location=device))
    model.eval()
    
    # Load FASTA and generate embedding ----------------------------------------
    emb, protein_id = generate_embedding_from_sequence(
        fasta_path=args.fasta,
        plm=args.model, 
        verbose=args.verbose,
        device=device
    )
    
    if args.verbose:
        print(f"Protein ID: {protein_id}")
        print(f"Sequence length: {emb.shape[1]} residues")
    
    # Predict ------------------------------------------------------------------
    if args.verbose:
        print(f"\nPredicting disorder (window={window_len}) ")
    centers, predictions = predict_sliding_window(
        model, emb, window_len, step=1, 
        use_softmax=config.get('soft_max', True),
        median_filter_size=None  # No smoothing
    )
    
    # Calculate disorder percentage
    stats = calculate_disorder_percentage(predictions, 
                                          threshold=threshold)
    
    # Print results
    print(f"DISORDER PREDICTION RESULTS FOR: {protein_id}")
    print(f"Total residues:        {stats['total_residues']}")
    print(f"Disordered residues:   {stats['disordered_residues']} (>{threshold} threshold)")
    print(f"Disorder percentage:   {stats['disorder_percentage']:.2f}%")
    
    # Generate plot if needed
    if args.plot:
        plot_output = Path(args.plot)
        plot_disorder_prediction(
            centers, 
            predictions, 
            protein_id, 
            threshold=threshold,
            output_path=plot_output
        )
    
    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            import pandas as pd
            df = pd.DataFrame({
                'position': centers,
                'structured_score': predictions[:, 0].numpy(),
                'disordered_score': predictions[:, 1].numpy(),
                'predicted_label': (predictions[:, 1] > threshold).numpy().astype(int)
            })
            df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
        else:
            print(f"\nUnknown output format: {output_path.suffix}")
            print("   Supported formats: .csv")
    
    return stats

if __name__ == '__main__':
    main()