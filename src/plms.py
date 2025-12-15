import os
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re 
import esm
import sys 


def get_esm2(fasta_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model directly from fair-esm package instead of torch.hub
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    num_repr_layer = 33  

    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Read all protein sequences from the FASTA file
    with open(fasta_path,"r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    sequences = ["".join(list(re.sub(r"[UZOB]", "X", str(record.seq))))  for record in records]
    protein_ids = [record.id for record in records]

    for prot_id, seq in tqdm(zip(protein_ids,sequences)):
        data = [(prot_id,seq)]          

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)
        # Extract per-residue representations (on CPU)
        with torch.no_grad(): # aca se cambio esto de los contactos
            results = model(batch_tokens, repr_layers=[num_repr_layer], return_contacts=False)
        
        for i , embed in enumerate(results["representations"][num_repr_layer]):
            # Extract embedding, remove special tokens (BOS and EOS)
            new_embed = embed.cpu().numpy()[1:batch_lens[0]-1]
            # Transpose to (emb_dim, L) format for consistency with ProtT5
            new_embed = new_embed.T
            np.save(os.path.join(output_dir,f'{prot_id}.npy') , arr=new_embed)
      
def compute_esmc_embed(sequence, model="esmc_300m", device="cuda"):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)

    # Load the pretrained ESMC model and move it to the specified device (GPU or CPU)
    client = ESMC.from_pretrained(model).to(device)

    # Encode the protein sequence into model-ready tensor format
    protein_tensor = client.encode(protein)

    # Run the model to obtain per-residue embeddings
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )

    return logits_output.embeddings

def get_esmc(fasta_path, output_dir, esmc_model):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read all protein sequences from the FASTA file
    with open(fasta_path, "r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    # Clean sequences (replace rare amino acids) and extract their IDs
    sequences = [
        "".join(list(re.sub(r"[UZOB]", "X", str(record.seq))))
        for record in records
    ]
    protein_ids = [record.id for record in records]

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        embedding = compute_esmc_embed(
            sequence=seq,
            model=esmc_model,
            device=device
        ).cpu().numpy()[0]

        batch_lens = embedding.shape[0]

        # Remove the first ([CLS]) and last ([EOS]) token embeddings
        embedding = embedding[1:batch_lens - 1]

        np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=embedding)


def get_ProtT5(fasta_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # Load the tokenizer and encoder model for ProtT5
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # Use full precision on CPU and half precision on GPU to save memory
    model.full() if device == 'cpu' else model.half()

    model = model.eval()

    # Read all sequences from the input FASTA file
    with open(fasta_path, "r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    for sliced_rec in tqdm([records[i:i+1] for i in range(0, len(records), 1)]):

        # Extract sequence IDs, sequences and lengths
        keys = [record.id for record in sliced_rec]
        sequence_examples = [str(record.seq) for record in sliced_rec]
        lens = [len(seq) for seq in sequence_examples]

        # Only process sequences shorter than 4000 residues
        if lens[0] < 4000:
            # Replace rare/ambiguous amino acids (U, Z, O, B) with 'X'
            # and insert spaces between residues (as required by ProtT5)
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

            # Tokenize sequences and pad to the longest sequence
            ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)

            numpy_embedding = embedding.last_hidden_state.cpu().numpy()

            for i, embed in enumerate(numpy_embedding):
                # Remove padding: keep only the first `lens[i]` embeddings
                new_embed = embed[:lens[i], :].T  # Transpose to [embedding_dim, sequence_length]

                # Save the embedding as a .npy file using the sequence ID
                np.save(os.path.join(output_dir, f'{keys[i]}.npy'), arr=new_embed)


def get_ProstT5(fasta_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the ProstT5 tokenizer and encoder model from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    # Use full precision on CPU and half precision on GPU
    model.full() if device == 'cpu' else model.half()

    model = model.eval()

    # Open and parse the FASTA file using Biopython
    with open(fasta_path, "r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))

    # Process each sequence individually (can be adapted for batch processing)
    for sliced_rec in tqdm([records[i:i+1] for i in range(0, len(records), 1)]):
        # Extract the sequence ID(s)
        keys = [record.id for record in sliced_rec]

        # Extract raw sequences as strings
        sequence_examples = [str(record.seq) for record in sliced_rec]

        # Get the lengths of each sequence
        lens = [len(seq) for seq in sequence_examples]

        # Replace rare amino acids with 'X' and insert spaces between residues
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

        # Add ProstT5-specific prompt token "<AA2fold>" at the start of each sequence
        sequence_examples = ["<AA2fold> " + s for s in sequence_examples]

        # Tokenize sequences using the ProstT5 tokenizer
        ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")

        # Convert tokenized inputs to PyTorch tensors and send to the appropriate device
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        # Move embeddings to CPU and convert to NumPy array
        numpy_embedding = embedding.last_hidden_state.cpu().numpy()

        # Process each embedding (in this case, only one per iteration)
        for i, embed in enumerate(numpy_embedding):
            # Remove the first token (corresponding to the <AA2fold> prompt)
            # Keep only the embeddings for actual amino acid residues
            new_embed = embed[1:lens[i]+1, :].T  # Transpose to [embedding_dim, sequence_length]

            # Save the resulting embedding as a .npy file named after the sequence ID
            np.save(os.path.join(output_dir, f'{keys[i]}.npy'), arr=new_embed)



