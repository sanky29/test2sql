from model import TextToSql
from dataloader import get_dataloader
from torchsummary import summary
from tqdm import tqdm 
from torch.nn.functional import one_hot
from config import dump_to_file
from torch.optim import Adam
import pdb
import torch
import os
import sys


class Trainer:

    def __init__(self,args):

        #store the args
        self.args = args

        #get the meta data
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.device = self.args.device
        self.init_dirs()
        self.steps = 0
        
        #validation and checkpoint frequency
        self.validate_every = self.args.validate_every
        self.checkpoint_every = self.args.checkpoint_every
        
        #get the model
        self.model = TextToSql(self.args).to(self.device)#, normalization)
        
        #the vocab info
        self.decoder_vocab_len = len(self.model.decoder.embeddings.vocab_dict)+1
        self.encoder_vocab = self.model.encoder_vocab
        self.decoder_vocab = self.model.decoder_vocab
        self.dbid_dict = self.model.decoder.dbid_dict
        
        #get the data
        self.train_data = get_dataloader(self.args, self.encoder_vocab, self.decoder_vocab, self.dbid_dict)
        self.val_data = get_dataloader(self.args, self.encoder_vocab, self.decoder_vocab, self.dbid_dict, mode = 'val')
        
        #the minimum validation loss
        self.optimizer = self.get_optimizer()
        self.min_validation_loss = sys.float_info.max 

    def init_dirs(self):

        #create the root dirs
        self.root = os.path.join(self.args.root, self.args.experiment_name)
        if(not os.path.exists(self.root)):
            os.makedirs(self.root)
        self.checkpoint_dir = os.path.join(self.root, 'checkpoint')
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.log_file = os.path.join(self.root, 'log.txt')
        with open(self.log_file, "w") as f:
            f.write('train_loss, val_loss\n')
            pass
        self.args_file = os.path.join(self.root, 'config.yaml')
        dump_to_file(self.args_file, self.args)
        
    def save_model(self, best = False):
        if(best):
            checkpoint_name = f'best.pt'
        else:
            checkpoint_name = f'model_{self.epoch_no}.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(self.model, checkpoint_path)
    
    def save_results(self, results):
        with open(self.log_file, "a") as f:
            s = ''
            for loss in results:
                s += str(loss) + ','
            f.write(f'{s}\n')
            f.close()

    def get_optimizer(self):
        rhos, params = self.get_parameters()
        params = [{"params": rhos, "lr": self.lr*0.005, "weight_decay":0.0},{'params': params}]
        return Adam(params, lr = self.lr, weight_decay=0.0001)
            
    def get_parameters(self):
        rhos = []
        non_rhos = []
        for name, param in self.model.named_parameters():
            print(name)
            if('encoder' in name):
                rhos.append(param)
            else:
                non_rhos.append(param)
        return rhos, non_rhos

    def print_results(self, results, validate = False):
        
        #print type of results
        print()
        if(validate):
            print("----VALIDITAION----")
        else:
            print(f'----EPOCH: {self.epoch_no}----')
        
        #print losses
        print(f'loss: {results}')
        print()
        print("-"*20)
    

    def loss(self, true_label, predicted_labels):
        
        #add loss
        predicted_labels = predicted_labels + 1e-15
        
        #output mask
        #mask: B x OSL x 1 
        mask = true_label.clone()
        mask[mask != 0] = 1
        mask = mask.unsqueeze(-1)

        #remove padding labels
        #true: B x OSL
        
        true_label = one_hot(true_label, num_classes = self.decoder_vocab_len)
        #loss: B x OSL x VOCAB_DICT  
        loss = -1*true_label*torch.log(predicted_labels)*mask
        return (loss.sum(-1).sum(-1)/mask.sum(-1).sum(-1)).mean()

    #validate function is just same as the train 
    def validate(self):
        
        #we need to return the average loss
        losses = []

        #with torch no grad
        with torch.no_grad():
            self.model.eval()

            #run for batch
            predicted_labels = None
            true_label = None
        
            with tqdm(total=len(self.val_data)) as t:   

                for db_id, input_sequence, output_sequence in self.val_data:
                
                    
                    #predict the labels and loss
                    #predicted_labels: [B x OSL-1 x DEOCDER_VOCAB]
                    predicted_labels = self.model(db_id, input_sequence, output_sequence)
                    
                    #cross entropy loss
                    loss = self.loss(output_sequence[:,1:].clone(), predicted_labels)

                    #append loss to the file
                    losses.append(loss.item())

                    #update the progress bar
                    t.set_postfix(loss=f"{loss:.7f}")
                    t.update(1)

                #print one result
                for db_id, input_sequence, output_sequence in self.val_data:
                    if(self.args.encoder.type == 'BERT'):
                        print(output_sequence[:1,:10])
                        print(torch.tensor(self.model(db_id[:1], input_sequence[:1])[:10]))
                    else:
                        print(output_sequence[:1,:10])
                        ind = (input_sequence[0] == 0).nonzero(as_tuple=True)[0][0]
                        print(torch.tensor(self.model(db_id[:1], input_sequence[:1,:ind])[:10]))
                        
                    break
                
        return sum(losses)/len(losses)


    def train_epoch(self):
        
        #we need to return the average loss
        losses = []

        #set in the train mode
        self.model.train()

        #run for batch
        with tqdm(total=len(self.train_data)) as t:   

            for db_id, input_sequence, output_sequence in self.train_data:
                
                # db = db_id[0]
                # isq = input_sequence[0]
                # osq = output_sequence[0]
                # pdb.set_trace()
                # for i in osq:
                #     print(i, self.model.decoder.db_mask[db,0,max(i-1,0)])
                # pdb.set_trace()
                # #zero out the gradient
                self.optimizer.zero_grad()

                #predict the labels and loss
                #predicted_labels: [B x OSL-1 x DEOCDER_VOCAB]
                predicted_labels = self.model(db_id, input_sequence, output_sequence)

                #cross entropy loss
                #output_sequence is dependency back in computation
                #graph
                loss = self.loss(output_sequence[:,1:].clone(), predicted_labels)

                #do backpropogation
                loss.backward()
                self.optimizer.step()

                #append loss to the file
                losses.append(loss.item())

                #update the progress bar
                t.set_postfix(loss=f"{loss:.2f}")
                t.update(1)     
            
        return sum(losses)/len(losses)

    def train(self):

        torch.autograd.set_detect_anomaly(True)
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch_no = epoch

            #the list which goes in the log text
            log = []
            
            #train for one epoch and print results
            train_loss = self.train_epoch()
            self.print_results(train_loss)
            log.append(train_loss)
            

            #checkpoint the model
            if((epoch + 1)%self.checkpoint_every == 0):
                self.save_model()

            #do validation if neccesary
            if((epoch + 1)% self.validate_every == 0):
                val_loss = self.validate()
                self.print_results(val_loss, validate = True)
                log.append(val_loss)