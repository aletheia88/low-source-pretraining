import torch
import math
import pandas as pd
import wandb
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from pathlib import Path

TOKENIZER_SAVEDIR = "tokenizer"
LM_MODEL_SAVEDIR = "bert-models-1"
VOCAB_SIZE = 256
MAX_LEN = 1320
MASKING_PROPORTION = 0.15
BATCH_SIZE = 4

class Dataset(Dataset):
    
    def __init__(self, df, tokenizer, max_length):
        self.examples = []
        # encode sequence into token ids
        for codewords in tqdm(df.values):
            words = codewords.split()
            chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
            for example in chunks:
                x = tokenizer.encode_plus(example,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True)
                assert(len(x.input_ids) == max_length)
                self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def train_LM(ds_name):
    
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_SAVEDIR,
                                                        max_len=MAX_LEN)
    train_dataset, valid_dataset = create_train_val_set(ds_name,
                                    tokenizer)
    data_collator = create_data_collector(tokenizer)
    create_train_bert_lm(data_collator, train_dataset, valid_dataset)

def fine_tune_LM(model_path):
    """Not sure how to fine-tune"""

    #model = RobertaForMaskedLM.from_pretrained(model_path)
    state_dict = torch.load('bert-models/pytorch_model.bin')

def create_train_val_set(file_name, tokenizer):

    df = pd.read_csv(file_name)
    train_df = df[df['is_valid'] == 0]
    valid_df = df[df['is_valid'] == 1]

    train_dataset = Dataset(train_df['codewords'], tokenizer, MAX_LEN)
    valid_dataset = Dataset(valid_df['codewords'], tokenizer, MAX_LEN)

    return train_dataset, valid_dataset

def create_data_collector(tokenizer):
    
    data_collator = DataCollatorForLanguageModeling(
                                    tokenizer=tokenizer,
                                    mlm=True,
                                    mlm_probability=MASKING_PROPORTION
                                    )
    return data_collator

def create_train_bert_lm(data_collator, train_dataset, valid_dataset):
    
    config = RobertaConfig(
                vocab_size=256,
                max_position_embeddings=MAX_LEN+2,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1
                )
    model = RobertaForMaskedLM(config=config)
    
    training_args = TrainingArguments(
                overwrite_output_dir=True,
                output_dir=LM_MODEL_SAVEDIR,
                num_train_epochs=10,
                per_device_train_batch_size=BATCH_SIZE,
                save_steps=1000,
                logging_steps=1000,
                evaluation_strategy="steps",
                eval_steps=1000,
                save_total_limit=1,
                prediction_loss_only=False,
                report_to="none"
                )

    trainer = Trainer(
                model=model,
                args = training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                eval_dataset=valid_dataset
                )
    
    trainer.train()
    
    trainer.save_model(LM_MODEL_SAVEDIR)

if __name__ == "__main__":
    
    train_LM('train_val_data.csv')

    ''' 
    model_path = 'bert-models'
    model = RobertaForMaskedLM.from_pretrained(model_path)
    parameters = model.parameters()

    for param in parameters:
        print(param)
        
    state_dict = torch.load('bert-models/pytorch_model.bin')
    print('state_dict')
    print(state_dict)
    '''
