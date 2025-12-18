# Protein Disorder Prediction Tool
This repository contains a deep learning tool for predicting intrinsically disordered regions (IDRs) in protein sequences. 

This tool generates embeddings from protein sequences using pre-trained protein language models (pLMs) and predicts disorder probabilities using a deep learning model with pre-trained weights. The output includes per-residue disorder scores, optional plots of disorder scores along the sequence and summary statistics.

## Environment setup

1. **Clone the repository:**
```bash
git clone https://github.com/sofiaaduarte/protein-disorder-tool.git
cd protein-disorder-tool
```

2. **Create a virtual environment**:
```bash
conda create -n protein-disorder-tool python=3.11
conda activate protein-disorder-tool
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Usage

The main script is `predict_disorder.py`. You can provide a FASTA file containing one or more protein sequences:

```bash
python predict_disorder.py --fasta data/samples.fasta
```

This script will:
- Read all sequences from the FASTA file.
- Generate embeddings using the specified pLM (ESM2 by default).
- Predict disorder scores for each residue using a sliding window approach.
- Save results (CSV and plots) to the output directory.
- Print disorder statistics to the console.

### Command-line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--fasta` | `-f` | Path to input FASTA file (Required). |
| `--model` | `-m` | Protein language model: `ESM2` (default) or `ProtT5`. |
| `--output-dir` | `-o` | Directory to save predictions (.csv) and plots (.png). Default: `results/`. |
| `--device` | `-d` | Device: `cpu`, `cuda` (default), `cuda:0`, etc. |
| `--verbose` | `-v` | Enable verbose output for detailed progress. Default: `False`. |

### Important Notes
- **ESM2 Sequence Limit**: The ESM2 model supports protein sequences up to 1024 residues. Any input exceeding this length will be truncated automatically, and a warning will be issued if this occurs.
- **Sequence preprocessing**: Non-canonical amino acids (U, Z, O, B) are automatically converted to 'X' before generating embeddings.
- **Device**: The tool runs on GPU (CUDA) by default for faster processing. Use the `--device cpu` flag to run the tool on a CPU.


### Examples

**1. Basic usage:**
```bash
python predict_disorder.py --fasta data/samples.fasta
```

**2. Specify output directory and verbose mode:**
```bash
python predict_disorder.py --fasta data/samples.fasta --output-dir my_results/ --verbose
```

**3. Use ProtT5 model on CPU:**
```bash
python predict_disorder.py --fasta data/samples.fasta --model ProtT5 --device cpu
```

**4. Use a specific GPU:**
```bash
python predict_disorder.py --fasta data/samples.fasta --device cuda:1
```

## Models

### Supported Protein Language Models

| Model | Description | Embedding Size |
|-------|-------------|----------------|
| **ESM2** | ESM-2 (650M parameters) | 1280 |
| **ProtT5** | ProtT5-XL (half precision) | 1024 |

The disorder prediction models are trained specifically for each pLM. 

Additional models will be added in future releases.