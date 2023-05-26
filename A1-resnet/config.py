from yacs.config import CfgNode

default_config = {
    'train_data': './../data/train.csv',
    'val_data': './../data/val.csv',
    'table_data':'./../data/tables.json',
    'data_root': './../data/',

    'experiment_name': 'demo',
    'root': './../experiments/',
    'resume_dir':None,
    'output_file':None,

    'glove_embedding': './../data/glove.6B.50d.txt',
    'encoder':'LSTM',
    'encoder':{
        'type':'BERT',
        'hidden_dim':128,
        'layers':1
    },
    'decoder':{
        'hidden_dim':128,
        'layers':1,
        'max_len':200,
        'k':10
    },
    'epochs':10000,
    'lr': 0.001,
    'device':'cuda',
    'checkpoint_every':1,
    'validate_every': 1,
    'checkpoint': None,
    'teacher_forcing_prob': 0.5,

    'train':{
        'batch_size':128
    },
    'test':{
        'batch_size':128
    },
    'val':{
        'batch_size':128
    },
    'val_split': None,

}
default_config = CfgNode(default_config)

#get the config from the extras
def get_config(extra_args):
    default_config.set_new_allowed(True)
    default_config.merge_from_list(extra_args)
    default_config.extras = extra_args
    return default_config


#load from a file
def load_config(file_name, new_config):
    
    #the file of args
    new_config.set_new_allowed(True)
    extras = new_config.extras
    new_config.merge_from_file(file_name)
    new_config.merge_from_list(extras)
    new_config.extras = []
    return new_config

#dump to file
def dump_to_file(file_name, config):
    config_str = config.dump()
    with open(file_name, "w") as f:
        f.write(config_str)
    