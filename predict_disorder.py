"""
Predict disorder from protein embeddings using a trained model.
"""
import argparse
import numpy as np
import torch as tr
from pathlib import Path
from src.model import BaseModel
from src.utils import ConfigLoader, load_embedding, predict_sliding_window, get_embedding_size
from src.plms import generate_embedding_from_sequence
from src.stats import calculate_disorder_percentage
from src.plot import plot_disorder_prediction
from Bio import SeqIO

def parser():
    parser = argparse.ArgumentParser(
        description='Predict disorder from protein embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input group, one can upload either an embedding, sequence, or fasta file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--embedding', '-e',
        type=str,
        help='Path to the embedding file (.npy)'
    )
    input_group.add_argument(
        '--sequence',
        type=str,
        help='Protein sequence as string (will generate embedding on-the-fly)'
    )
    input_group.add_argument(
        '--fasta', '-f',
        type=str,
        help='Path to FASTA file (will generate embedding on-the-fly)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        # required=True,
        default='/home/sduarte/IDPfun2/results/secondment/ESM2_filt150_ker9_resnet1_win36_lr1e-05_09-12-2025_00-04-07',
        help='Path to the model directory (containing weights.pk and config.yaml)'
    )
    parser.add_argument(
        '--step', '-s',
        type=int,
        default=1,
        help='Step size for sliding window'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Threshold for classifying residues as disordered'
    )   
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file to save predictions (.npy or .csv)'
    )
    parser.add_argument(
        '--plot', '-p',
        type=str,
        default=None,
        help='Output file to save plot (.png or .pdf)'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        help='Smoothing window size for plot (default: 3, 0 = no smoothing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run predictions on'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()



def main():
    args = parser()

    # Set up model path, config and weights ------------------------------------
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    config_path = model_dir / 'config.yaml'
    weights_path = model_dir / 'weights.pk'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load configuration -------------------------------------------------------
    if args.verbose:
        print(f"Model directory: {model_dir}")
        print(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()
    window_len = config.get('win_len', 13) # window length
    
    # Initialize model ---------------------------------------------------------
    if args.verbose:
        print(f"Loading model from: {weights_path}")
    categories = ['structured', 'disordered']
    model = BaseModel(
        len(categories),
        lr=config['lr'],
        device=args.device,
        emb_size=get_embedding_size(config.get('plm', 'esm2')),
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['n_resnet']
    )
    model.load_state_dict(tr.load(weights_path, map_location=args.device))
    model.eval()
    
    # Load or generate embedding -----------------------------------------------
    if args.embedding:
        # Load pre-computed embedding
        if args.verbose:
            print(f"Loading embedding from: {args.embedding}")
        emb = load_embedding(args.embedding)
        protein_id = Path(args.embedding).stem
    
    elif args.sequence:
        # Generate embedding from sequence
        protein_id = "sequence"
        emb = generate_embedding_from_sequence(args.sequence, protein_id, args.verbose)
    
    elif args.fasta:
        # Generate embedding from FASTA file
        if args.verbose:
            print(f"Reading FASTA file: {args.fasta}")
        
        # Read first sequence from FASTA
        with open(args.fasta, 'r') as f:
            records = list(SeqIO.parse(f, "fasta"))
        
        if not records:
            raise ValueError(f"No sequences found in FASTA file: {args.fasta}")
        
        if len(records) > 1 and args.verbose:
            print(f"Warning: Multiple sequences found, using only the first one: {records[0].id}")
        
        protein_id = records[0].id
        sequence = str(records[0].seq)
        emb = generate_embedding_from_sequence(sequence, protein_id, args.verbose)
    
    else:
        raise ValueError("No input provided. Use --embedding, --sequence, or --fasta")
    
    print(f"\nEmbedding shape: {emb.shape}")
    print(f"Protein ID: {protein_id}")
    print(f"Sequence length: {emb.shape[1]} residues")
    
    # Predict ------------------------------------------------------------------
    print(f"\nPredicting disorder (window={window_len}, step={args.step})...")
    centers, predictions = predict_sliding_window(
        model, emb, window_len, step=args.step, 
        use_softmax=config.get('soft_max', True),
        median_filter_size=args.smooth if args.smooth > 0 else None
    )
    
    # Calculate disorder percentage
    stats = calculate_disorder_percentage(predictions, threshold=args.threshold)
    
    # Print results
    print(f"DISORDER PREDICTION RESULTS FOR: {protein_id}")
    print(f"Total residues:        {stats['total_residues']}")
    print(f"Disordered residues:   {stats['disordered_residues']} (>{args.threshold} threshold)")
    print(f"Disorder percentage:   {stats['disorder_percentage']:.2f}%")
    
    # Generate plot if needed
    if args.plot:
        plot_output = Path(args.plot)
        plot_disorder_prediction(
            centers, 
            predictions, 
            protein_id, 
            threshold=args.threshold,
            output_path=plot_output
        )
    
    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.npy':
            np.save(output_path, predictions.numpy())
            print(f"\nPredictions saved to: {output_path}")
        
        elif output_path.suffix == '.csv':
            import pandas as pd
            df = pd.DataFrame({
                'position': centers,
                'structured_score': predictions[:, 0].numpy(),
                'disordered_score': predictions[:, 1].numpy(),
                'predicted_label': (predictions[:, 1] > args.threshold).numpy().astype(int)
            })
            df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
        
        else:
            print(f"\nUnknown output format: {output_path.suffix}")
            print("   Supported formats: .npy, .csv")
    
    return stats

if __name__ == '__main__':
    main()