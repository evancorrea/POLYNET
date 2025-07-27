import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from src.utils.encoding import one_hot

class PasDataset(Dataset):
    def __init__(self, fasta_paths, labels):
        """
        fasta_paths: list of FASTA file paths
        labels: corresponding labels [1,0]
        """
        self.seqs = []
        self.ys = []
        
        #loop through fasta paths and associated labels -> parse each fasta for 201nt segments -> onehot encode and append to seqs.
        for path, y in zip(fasta_paths, labels):
            for rec in SeqIO.parse(path, "fasta"):
                seq = str(rec.seq)
                if len(seq) != 201:
                    print('skipping sequence that is not 201bp')
                    continue
                self.seqs.append(one_hot(str(rec.seq)))
                self.ys.append(y)
                                 
        self.seqs = torch.tensor(self.seqs)
        self.ys = torch.tensor(self.ys, dtype=torch.float32)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.seqs[i], self.ys[i]
            