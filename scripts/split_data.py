pos = list(SeqIO.parse('../data/pos_201_hg19.fa', 'fasta'))
neg = list(SeqIO.parse('../data/neg_201_hg19.fa', 'fasta'))
random.seed(42)
random.shuffle(pos)
random.shuffle(neg)

def splits(lst, train_frac = 0.7, val_frac = 0.15):
    #splits the list of sequences based on index corresponding to set size. 
    n = len(lst)
    i_train = int(train_frac * n)
    i_val = i_train + int(val_frac * n)
    return lst[:i_train], lst[i_train:i_val], lst[i_val:]

pos_train, pos_val, pos_test = splits(pos)
neg_train, neg_val, neg_test = splits(neg)

#create separate fasta files for each set
from pathlib import Path

def write_split(records, path):
    Path(path).parent.mkdir(exist_ok = True)
    SeqIO.write(records, path, 'fasta')

write_split(pos_train, '../data/processed/pos_201_train.fa')
write_split(pos_val,   '../data/processed/pos_201_val.fa')
write_split(pos_test,  '../data/processed/pos_201_test.fa')

write_split(neg_train, '../data/processed/neg_201_train.fa')
write_split(neg_val,   '../data/processed/neg_201_val.fa')
write_split(neg_test,  '../data/processed/neg_201_test.fa')