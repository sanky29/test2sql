import os
import json
import pdb 
import csv
import re
CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'by', 'distinct']
JOIN_KEYWORDS = ['join', 'on', 'as']
WHERE_OPS = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists']
UNIT_OPS = ['none', '-', '+', "*", '/']
AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
SQL_OPS = ['intersect', 'union', 'except', 'having']
ORDER_OPS = ['desc', 'asc']
SPL_CHARS = ['(', ')', '.', '[', ']', ',']
ALL_TOKENS = [CLAUSE_KEYWORDS, JOIN_KEYWORDS, WHERE_OPS, UNIT_OPS, AGG_OPS, COND_OPS, SQL_OPS, ORDER_OPS, SPL_CHARS]

class Preprocessor:

    def __init__(self, args, mode = 'train'):
        self.args = args
        self.decoder_vocab = os.path.join(args.data_root, 'decoder_vocab.txt')
        self.encoder_vocab = os.path.join(args.data_root, 'encoder_vocab.txt')
        self.processed_val_decoder = os.path.join(args.data_root, 'processed_val_decoder.txt')
        self.processed_val_encoder = os.path.join(args.data_root, 'processed_val_encoder.txt')
        self.processed_train_decoder = os.path.join(args.data_root, 'processed_train_decoder.txt')
        self.processed_train_encoder = os.path.join(args.data_root, 'processed_train_encoder.txt')
        self.masks = os.path.join(args.data_root, 'masks.txt')


    def init_files(self):
        with open(self.decoder_vocab,'w') as f:
            pass
        with open(self.encoder_vocab,'w') as f:
            pass
        with open(self.processed_val_decoder,'w') as f:
            pass
        with open(self.processed_val_decoder,'w') as f:
            pass
        with open(self.processed_train_decoder,'w') as f:
            pass
        with open(self.processed_train_decoder,'w') as f:
            pass
        with open(self.masks,'w') as f:
            pass
        
    '''
    This is the most important function of all
    I would replace the column values in the query and 
    natural language to predefined token
    The function at the end will produce
        1. decoder_vocab.txt
        2. encoder_vocab.txt
        3. processed_decoder.txt
        4. processed_encoder.txt
        5. output_masks.txt
    '''

    def parse_table(self):

        #get the file
        schema = open(self.args.table_data, 'r')
        schema = json.load(schema)
        
        #the database ids
        self.databases_tokens = dict()
        for database in schema:
            dbid = database['db_id'].lower()
            self.databases_tokens[dbid] = set()
            
            #add columns
            for col in database['column_names_original']:
                self.databases_tokens[dbid].add(col[1].lower())
            
            #add tables
            for table in database['table_names_original']:
                self.databases_tokens[dbid].add(table.lower())
        self.schema = schema

    #generate sql vocab
    def generate_sql_vocab(self):
        
        #the vocab dictionary
        ret = dict()
        token_id = 0

        #iterate over predefined tokens
        for operator_set in ALL_TOKENS:
            for operator in operator_set:
                if(operator not in ret):
                    ret[operator] = token_id
                    token_id += 1

        #add the value and reference values
        for i in range(1,6):
            ret[f't{i}'] = token_id
            token_id += 1
            ret[f'VALUE_{i-1}'] = token_id
            token_id += 1
            ret[f'{i}'] = token_id
            token_id += 1
            
        
        #the same token for all tables
        self.comman_sql_tokens = token_id

        #add the databse specific tokens
        for database in self.databases_tokens:
            for database_token in self.databases_tokens[database]:
                if(database_token not in ret):
                    ret[database_token] = token_id
                    token_id += 1
        return ret

    '''
    split the data line
    '''
    def get_sql_tokens(self, sql, value_mapping):
        
        #lower the string
        sql = sql.lower()
        for i in value_mapping:
            sql = re.sub(rf'[\'\"]? ?\b{i}\b ?[\'\"]?', f' {value_mapping[i]} ', sql)
        sql_tokens = sql.replace('.', ' . ')
        sql_tokens = sql_tokens.replace('*', ' * ').replace('(', ' ( ')
        sql_tokens = sql_tokens.replace(')', ' ) ')
        sql_tokens = sql_tokens.split(' ')
        sql_tokens = [i.strip() for i in sql_tokens if i.strip() != ''] 
        sql_tokens =  ['<SOS>'] + sql_tokens + ['<EOS>']
        return sql_tokens

   
    #get the date
    def get_datetime(self,line):
        #this matches the dd/mm/yyyy type dates        
        dates_1 = re.findall('[\"\']?(\d+[-./|]\d+[-./|]\d+)[\"\']?', line)
        
        #this matches dd/mm or mm/yyyy type dates
        dates_2 = re.findall('[\"\']?(\d+[-./|]\d+)[\"\']?', line)
        
        #time hh:mm:ssAM or PM
        time_1 = re.findall('[\"\']?(\d+[-./|:]\d+[-./|:]\d+(?:|am|pm))[\"\']?', line)
        time_2 = re.findall('[\"\']?(\d+[-./|:]\d+(?:|am|pm))[\"\']?', line)
        
        return dates_1 + time_1 + dates_2 + time_2

    #get just integers
    def get_numbers(self, line):
        numbers = re.findall('[\"\']?(\d+(?:|th|nd))[\"\'.]?', line)
        return numbers
    
    #def special words
    def get_words(self, line):
        #single words like 'sanket' or " sanket "
        words_1 = re.findall('\" ?([\w+ ?]+) ?\"', line)
        words_2 = re.findall('\' ?([\w+ ?]+) ?\'', line)
        words_3 = re.findall('\' ?([\w. ?]+\w*) ?\'', line)
        
        return words_1 + words_2 + words_3

    '''
    split the data line
    '''
    def get_text_tokens(self, line):
        
        #lower the string
        line = line.lower()
        line = line.replace("?", " ? ")
        
        words = self.get_words(line)
        numbers = self.get_datetime(line) + self.get_numbers(line)
        possible_values = words + numbers
        
        value_map = dict()
        values = 0
        for v in possible_values:
            new_line = line.replace(v, f'VALUE_{values}')
            if(new_line != line):
                value_map[v] = f'VALUE_{values}'
                values += 1
            line = new_line

        #split on spaces
        tokens = line.split(' ')
        tokens = [i.strip() for i in tokens]
        return tokens, value_map

    def preprocess(self):

        '''do preprocessiong on combined data 
            from train.csv and dev.csv
        '''
        #init the directories
        self.init_files()
        
        #parse the table.json as
        self.parse_table()

        #first generate sql_query vocab
        self.sql_vocab = self.generate_sql_vocab()
        
        #open file
        text_file = open(self.processed_train_encoder, "w")
        sql_file = open(self.processed_train_decoder, "w")

        j = 0
        #generate tokenized sentence
        data_file = open(self.args.train_data, 'r')
        data_file = csv.reader(data_file)
        for line in data_file:
            text_tokens, value_mapping = self.get_text_tokens(line[2])
            sql_tokens = self.get_sql_tokens(line[1].lower(), value_mapping)
                    
            # for t in sql_tokens:
                # if(t not in self.sql_vocab):
                #     j += 1
                #     print(' '.join(sql_tokens))
                #     print(line[2])
                #     print(t)
                #     print()
                #     break
            #write to file
            text_file.write(line[0] + ', '+ ' '.join(text_tokens) + "\n")
            sql_file.write(' '.join(sql_tokens) + "\n")
        
        #open file
        text_file = open(self.processed_val_encoder, "w")
        sql_file = open(self.processed_val_decoder, "w")
            
        #generate tokenized sentence
        data_file = open(self.args.val_data, 'r')
        data_file = csv.reader(data_file)
        for line in data_file:
            text_tokens, value_mapping = self.get_text_tokens(line[2])
            sql_tokens = self.get_sql_tokens(line[1], value_mapping)

            #write to file
            text_file.write(line[0] + ', ' + ' '.join(text_tokens) + "\n")
            sql_file.write(' '.join(sql_tokens) + "\n")

        outfile = open(self.decoder_vocab, "w")
        for i in self.sql_vocab:
            outfile.write(i + "\n")

        outfile = open(self.masks, "w")
        non_common_token = len(self.sql_vocab) - self.comman_sql_tokens
        for database in self.schema:
            
            mask = [1 for i in range(0, (self.comman_sql_tokens))]+ [0 for i in range(0, non_common_token)]
            for col in database['column_names_original']:
                mask[self.sql_vocab[col[1].lower()]] = 1
            
            #add tables
            for table in database['table_names_original']:
                mask[self.sql_vocab[table.lower()]] = 1
            
            mask = [str(i) for i in mask]
            mask = [database['db_id']] + mask
            outfile.write(' '.join(mask) + "\n")


        