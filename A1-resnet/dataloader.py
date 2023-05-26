import pickle
import pdb
import torch 
import numpy as np 
import os
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
class Data:

    def __init__(self, args, encoder_vocab, decoder_vocab, dbid_dict, mode = 'train'):

        #read the files
        encoder_file = os.path.join(args.data_root, f'processed_{mode}_encoder.txt')
        decoder_file = os.path.join(args.data_root, f'processed_{mode}_decoder.txt')
        
        self.encoder_seq = []
        self.decoder_seq = []
        self.database_id = []
        
        f = open(encoder_file, 'r')
        for line in f.readlines():
            data = line.split(',')
            self.database_id.append(dbid_dict[data[0]])
            
            wl = []
            for word in data[1].strip().split(' '):
                wl.append(encoder_vocab(word))
            self.encoder_seq.append(wl)

        f = open(decoder_file, 'r')
        for line in f.readlines():
            wl = []
            for word in line.strip().split(' '):
                wl.append(decoder_vocab(word))
            self.decoder_seq.append(wl)

    def __getitem__(self, index):
        return self.database_id[index], self.encoder_seq[index], self.decoder_seq[index]

    def __len__(self):
        return len(self.encoder_seq)

class DataBERT:

    def __init__(self, args, encoder_vocab, decoder_vocab, dbid_dict, mode = 'train'):

        #read the files
        encoder_file = os.path.join(args.data_root, f'processed_{mode}_encoder.txt')
        decoder_file = os.path.join(args.data_root, f'processed_{mode}_decoder.txt')
        
        self.encoder_seq = []
        self.decoder_seq = []
        self.database_id = []
        
        f = open(encoder_file, 'r')
        for line in f.readlines():
            data = line.split(',')
            self.database_id.append(dbid_dict[data[0]])
            self.encoder_seq.append(data[1].strip())

        f = open(decoder_file, 'r')
        for line in f.readlines():
            wl = []
            for word in line.strip().split(' '):
                wl.append(decoder_vocab(word))
            self.decoder_seq.append(wl)

    def __getitem__(self, index):
        return self.database_id[index], self.encoder_seq[index], self.decoder_seq[index]

    def __len__(self):
        return len(self.encoder_seq)


# #the collat function
def collate_fn_lstm(batch):
    input_sequence = pad_sequence([torch.tensor(t[1]) for t in batch], batch_first = True).long().cuda()
    output_sequence = pad_sequence([torch.tensor(t[2]) for t in batch], batch_first = True).long().cuda()
    database_id = torch.tensor([t[0] for t in batch]).cuda()
    return database_id, input_sequence, output_sequence

# #the collat function
def collate_fn_bert(batch):
    input_sequence = [t[1] for t in batch]
    output_sequence = pad_sequence([torch.tensor(t[2]) for t in batch], batch_first = True).long().cuda()
    database_id = torch.tensor([t[0] for t in batch]).cuda()
    return database_id, input_sequence, output_sequence


def get_dataloader(args, encoder_vocab, decoder_vocab, dbid_dict, mode = 'train'):

    shuffle = True
    if(mode == 'val'):
        shuffle = False
    if(args.encoder.type == 'BERT'):
        data = DataBERT(args, encoder_vocab, decoder_vocab, dbid_dict, mode)
        data = DataLoader(data, batch_size = 4, shuffle = shuffle, collate_fn = collate_fn_bert)
    else:
        data = Data(args, encoder_vocab, decoder_vocab, dbid_dict, mode)
        data = DataLoader(data, batch_size = 4, shuffle = shuffle, collate_fn = collate_fn_lstm)
    return data