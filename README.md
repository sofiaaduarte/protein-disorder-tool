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

The main script is `predict_disorder.py`. You need to provide a FASTA file:

```bash
python predict_disorder.py --fasta data/sample.fasta
```

This script will:
- Read the first sequence from the FASTA file
- Generate embeddings using ESM2 (default model)
- Predict disorder scores for each residue
- Print disorder statistics to the console

This are the available command-line arguments:

| Argument | Short | Description |
|----------|-------|-------------|
| `--fasta` | `-f` | Path to input FASTA file (required) |
| `--model` | `-m` | Protein language model: `ESM2` (default) or `ProtT5` |
| `--output` | `-o` | Output CSV file with predictions |
| `--plot` | `-p` | Output plot file (`.png` or `.pdf`) |
| `--device` | `-d` | Device: `cpu`, `cuda` (default), `cuda:0`, `cuda:1`, etc. |
| `--verbose` | `-v` | Enable verbose output |

### Examples

**1. Generate a disorder plot:**
```bash
python predict_disorder.py --fasta data/sample.fasta --plot results/sample_plot.png
```

**2. Save predictions to CSV:**
```bash
python predict_disorder.py --fasta data/sample.fasta --output results/sample_predictions.csv
```

**3. Use ProtT5 model:**
```bash
python predict_disorder.py --fasta data/sample.fasta \
    --model ProtT5 \
    --plot results/sample_plot.png
```

**4. Use specific GPU (e.g., second GPU) or run on CPU:**
```bash
python predict_disorder.py --fasta data/sample.fasta \
    --device cuda:1 \
    --plot results/sample_plot.png
```

```bash
python predict_disorder.py --fasta data/sample.fasta \
    --device cpu \
    --output results/sample_predictions.csv
```

**6. Full pipeline with all outputs:**
```bash
python predict_disorder.py --fasta data/sample.fasta \
    --model ESM2 \
    --output results/sample_predictions.csv \
    --plot results/sample_plot.pdf \
    --device cuda:0 \
    --verbose
```

## Output

### Console Output
The tool prints disorder statistics to the console. For example, for the sample protein `DP03212`:
```
DISORDER PREDICTION RESULTS FOR: DP03212
Total residues:        419
Disordered residues:   254 (>0.5 threshold)
Disorder percentage:   60.62%
```

### CSV Output (`--output`)
When using `--output`, a CSV file is generated with the following columns:
- `position`: Residue position
- `structured_score`: Probability of structured region
- `disordered_score`: Probability of disordered region
- `predicted_label`: Binary prediction (0=structured, 1=disordered)

### Plot Output (`--plot`)
When using `--plot`, a visualization is generated showing:
- Disorder propensity along the sequence
- Threshold line (default: 0.5)
- Disordered regions highlighted

## Models

## Model Performance
Working on this section.


### Supported Protein Language Models

| Model | Description | Embedding Size |
|-------|-------------|----------------|
| **ESM2** | ESM-2 (650M parameters) | 1280 |
| **ProtT5** | ProtT5-XL (half precision) | 1024 |

The disorder prediction models are trained specifically for each pLM. 

Additional models will be added in future releases.

### Model Architecture
Working on this section.


## Input Format
The tool expects as input a protein sequence in a FASTA format. If the FASTA file contains multiple sequences, only the first sequence is processed. Unusual or non-canonical amino acids, including U, Z, O and B, are automatically converted to X before embedding generation.

## Performance
Time, memory?