import os
import logging
import argparse
from Bio import SeqIO
import src.plms as plms

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', 
                        help='Input fasta file with sequences to embed')
    parser.add_argument('--model_name',
                        help='ProtT5, ProstT5, esm2, esmc_300m, esmc_600m')
    parser.add_argument('--root',
                        help='the directory that the embeddings must be stored')
    parser.add_argument('--log_file',
                        help='log file to record the embedding process')

    args = parser.parse_args()
    os.makedirs(args.root , exist_ok=True)
    return args 

if __name__ == '__main__':
    args = parser()

    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
    logging.info(f'================= Generate Embeddings for {args.model_name}  =================')
    
    # ------- Check if the embeddings exist, don't calculate them again ------ #
    root_files = [f.split('.')[0] for f in os.listdir(args.root)]
    
    with open(args.input_fasta,"r", encoding="utf-8") as handle:
        protein_ids = [record.id for record in list(SeqIO.parse(handle, "fasta"))]

    logging.info(set(protein_ids) - set(root_files))
    if set(protein_ids) in set(root_files) :
        logging.info(f'Embedding for {args.model_name} already exist in {args.root}')
    
    # -------------------------- Compute embeddings -------------------------- #
    else:
        logging.info(f'Computing embeddings for {args.model_name}') 

        if args.model_name == 'ProtT5':
            plms.get_ProtT5(fasta_path=args.input_fasta, output_dir=args.root)
        
        elif args.model_name == 'ProstT5':
            plms.get_ProstT5(fasta_path=args.input_fasta, output_dir=args.root)
        
        elif args.model_name == 'esm2':
            plms.get_esm2(fasta_path=args.input_fasta, output_dir=args.root)
            
        elif args.model_name in ['esmc_300m','esmc_600m']:
            plms.get_esmc(fasta_path=args.input_fasta, output_dir=args.root, esmc_model=args.model_name)

        logging.info(f'Embeddings for {args.model_name} are stored in {args.root}')
        logging.info(f'================= Finished Embeddings for {args.model_name}  =================')
    