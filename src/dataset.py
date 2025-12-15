from torch.utils.data import Dataset
import torch as tr
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class SegmentDataset(Dataset): 
    """
    A PyTorch Dataset class for sampling regions of proteins with multiple domains
    or annotated regions. This dataset samples ONE fixed-size window from each
    protein region for training or evaluation.
    """
    def __init__(self, dataset_path, emb_path, categories=("structured", "disordered"),
                 win_len=32, debug=False, is_training=False):
        """
        Dataset contains all valid segments in the complete proteins.
        """
        self.is_training = is_training
        self.emb_path = emb_path
        self.categories = categories
        self.win_len = win_len
        self.dataset = pd.read_csv(dataset_path)

        # If debugging, sample a smaller subset of the dataset
        if debug:
            self.dataset = self.dataset.sample(n=100)

    def __len__(self):
        return len(self.dataset)

    def soft_domain_score(self, window_start, window_end, domain_start, domain_end):
        """
        Calculate the fraction of a fixed-size window that is covered by a domain interval.
        This returns the proportion (a float between 0.0 and 1.0) of the interval
        [window_start, window_end) that overlaps with [domain_start, domain_end).
        """
        return max(0, (min(domain_end, window_end) - max(domain_start, window_start))/(window_end-window_start))
        # IMPROVE: make this easier to read and understand

    # Reduced cache size to balance performance and memory usage
    @lru_cache(maxsize=32)
    def _load_emb(self, acc):
        """Load precomputed embedding for a given protein accession."""
        emb = np.load(f"{self.emb_path}{acc}.npy")
        # emb = emb.T # ! Ensure format is (emb_dim, L), not (L, emb_dim)
        return tr.tensor(emb, dtype=tr.float32)

    def __getitem__(self, item):
        """Sample one random or centered window from a region entry"""
        n_item = item
        item = self.dataset.iloc[item]

        emb = self._load_emb(item.acc)
        
        # Ensure the domain range is valid (start is less than end)
        if item.start >= item.end:
            # Skip invalid domain ranges and try the next sample
            return self.__getitem__((int(n_item) + 1) % len(self.dataset))

        # Determine the center of the window
        if self.is_training:
            # Randomly sample the center for training
            center = np.random.randint(item.start, item.end)
        else:
            # Use the midpoint for evaluation
            center = (item.start + item.end)//2

        # Calculate the start and end of the window
        start = max(0, center - self.win_len//2)
        end = min(emb.shape[1], center + self.win_len//2)

        # Create a hard label based on the current item's label
        label_hard = self.categories.index(item.label)

        # Initialize the label vector. It's a soft label because it represents
        # the coverage of different classes in the window.
        label_soft = tr.zeros(len(self.categories))

        # Get all the domains/regions for the current protein
        domains = self.dataset[self.dataset.acc==item.acc]
        
        # Calculate coverage of the window on each domain to get a class score
        for k in range(len(domains)):
            score = self.soft_domain_score(start, end, domains.iloc[k].start, domains.iloc[k].end)
            label_ind = self.categories.index(domains.iloc[k].label)
            label_soft[label_ind] += score
            # label[label_ind] = max(score, label[label_ind]) # * this was used in emb2pfam

        # # force labels to sum 1 # * this was used in emb2pfam
        # s = label.sum()
        # if s<1: 
        #     ind = tr.where(label==0)[0]
        #     label[ind] = (1-s)/len(ind)

        # Create a fixed-size embedding window
        emb_win = tr.zeros((emb.shape[0], self.win_len), dtype=tr.float)
        emb_win[:,:end-start] = emb[:, start:end]
        
        # Ensure tensors are contiguous for DataLoader with multiple workers
        emb_win = emb_win.contiguous()
        label_soft = label_soft.contiguous()

        return emb_win, label_soft, label_hard, center, item.acc, start, end
    

class AminoAcidDataset(Dataset):
    """
    This dataset samples a window around each annotated residue in a protein, to
    simulate a sliding window approach for evaluation.
    """
    def __init__(self, dataset_path, emb_path, categories = ("structured", "disordered"),
                 win_len=32, step=1, debug=False):
    
        self.win_len = win_len
        self.half_win = win_len // 2
        self.emb_path = Path(emb_path)
        self.categories = categories
        self.cat2idx = {c: i for i, c in enumerate(categories)}  # Map category (state) names to indices

        # Load the dataset
        df = pd.read_csv(dataset_path).astype({"start": int, "end": int})

        if debug: # to debug, select only one acc randomly
            # df = df[df.acc == "P37268"] 
            df = df.sample(n=1)
            
        # Get the domains/regions for each protein (acc)
        self.domains = {}
        for acc, g in df.groupby("acc"):
            self.domains[acc] = [
                # Store start, end, and label index
                (int(r.start), int(r.end), self.cat2idx[r.label])
                for r in g.itertuples(index=False)
            ]

        # Preload the lengths of protein embeddings (emb_dim, L)
        prot_len = {
            acc: np.load(self.emb_path / f"{acc}.npy", mmap_mode="r").shape[1]
            for acc in self.domains.keys()
        }

        # Build a list of valid center positions
        examples = []
        for acc, doms in self.domains.items():
            L = prot_len[acc] 

            for start, end, label in doms:  # Iterate over domains
                for c in range(start, end + 1, step): # Include the end value in the range
                    start = max(0, c - self.half_win)
                    end   = min(L, c + self.half_win)

                    if end - start > 0:  # Only include valid windows
                        examples.append((acc, c, label))

        if not examples:
            raise RuntimeError("No valid centres found: revise win_len or annotations")

        # Sort examples by protein accession and center position
        self.examples = sorted(examples, key=lambda t: (t[0], t[1]))

    def __len__(self):
        # Number of valid amino acids centers ("examples")
        return len(self.examples)  
    
    # Enable cache with limited size to improve performance
    @lru_cache(maxsize=32)
    def _load_emb(self, acc):
        # Load the precomputed embedding for a given protein accession
        emb = np.load(self.emb_path / f"{acc}.npy")
        # emb = emb.T # ! Ensure format is (emb_dim, L), not (L, emb_dim)
        return emb

    def _soft_score(self, acc, win_start, win_end):
        """Compute soft scores for the window based on domain overlap."""
        scores = np.zeros(len(self.categories), dtype=np.float32)
        for dom_start, dom_end, label in self.domains[acc]:  # Iterate over domains for the protein
            overlap = max(0, min(dom_end, win_end) - max(dom_start, win_start))  # Calculate overlap
            if overlap:
                scores[label] += overlap / (win_end - win_start)  # Normalize by window size
        return scores

    def __getitem__(self, idx):
        """Sample one window from a region entry"""
        # Get the amino acid example details
        acc, center, label_hard = self.examples[idx]  
        emb = self._load_emb(acc) 
        L = emb.shape[1]  

        # Calculate window boundaries
        half = self.win_len // 2
        start = max(0, center - half)
        end   = min(L, center + half)

        # Compute soft scores for the window and convert to tensor
        label_soft = self._soft_score(acc, start, end)  
        label_soft = tr.tensor(label_soft, dtype=tr.float32)

        # Create a fixed-size embedding window
        seg_len = end - start
        win = np.zeros((emb.shape[0], self.win_len), dtype=emb.dtype) 
        win[:, :seg_len] = emb[:, start:end]
        win = tr.tensor(win, dtype=tr.float32)

        return win, label_soft, label_hard, center, acc, start, end
    
# IMPROVE: Improve the AminoAcidDataset class to make it easier to read and understand