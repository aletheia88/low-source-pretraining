import torch
import torch.nn as nn
import torch.nn as nn

import math
import pandas as pd
from transformers import RobertaForMaskedLM
from tqdm import tqdm
from train_bert_lm import *
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import TensorDataset, DataLoader, RandomSampler,\
                                                        SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

class genreClassifier(nn.Module):

    def __init__(self, freeze_bert=False):
        
        super(genreClassifier, self).__init__()
        D_in, D_out = 768, 10

        self.bert = RobertaForMaskedLM.from_pretrained('bert-models')
        
        self.classifier = nn.Sequential(
                        nn.Linear(D_in, D_out)
                        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        
        # curr outputs[1][-1] shape - torch.Size([4, 1320, 768])
        # outputs shape from tutorial - torch.Size([32, 64, 768])
        
        last_hidden_state_cls = outputs[1][-1][:, 0, :]
        #last_hidden_state_cls = outputs[0][:, 0, :]
        
        #print(f'last_hidden_state_cls: {last_hidden_state_cls.shape}')
        
        logits = self.classifier(last_hidden_state_cls)

        return logits

def train(train_dataloader, valid_dataloader, epochs):
    
    device = torch.device(0)
    loss_fn = nn.CrossEntropyLoss()
    
    model = genreClassifier(freeze_bert=False)
    model.to(device)

    optimizer = AdamW(model.parameters(),
                        lr=5e-5,
                        eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    for epoch_i in range(epochs):
        
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            #print(f'id: {b_input_ids.shape}, mask: {b_attn_mask.shape}, labels: {b_labels.shape}')
            
            logits = model(b_input_ids, b_attn_mask)
            #print(f'logits: {logits}')
            #print(f'logits shape: {logits.shape}')
            
            #print(f'b_labels shape: {b_labels.shape}')
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()


def create_dataloader(input_ids, attention_masks, labels, batch_size=4):
    
    ## encode labels ##
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_labels = pd.DataFrame(encoder.fit_transform(labels).toarray()).values
    
    encoded_labels = torch.tensor(encoded_labels)
     
    data = TensorDataset(input_ids, attention_masks, encoded_labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
    
def get_input_ids_att_masks(data_df):

    tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer') 
     
    input_ids = []
    attention_masks = []
    max_length = 1320

    for codewords in tqdm(data_df.values):
        
        words = codewords.split()
        chunks = [' '.join(words[i:i+max_length])\
                            for i in range(0, len(words), max_length)]
        for example in chunks:
            x = tokenizer.encode_plus(example,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True)
        
            input_ids.append(x.get('input_ids'))
            attention_masks.append(x.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
        
    return input_ids, attention_masks

if __name__ == "__main__":
    
    df = pd.read_csv('train_val_data.csv')
    train_df = df[df['is_valid'] == 0]['codewords'] 
    valid_df = df[df['is_valid'] == 1]['codewords']
    
    train_labels = df[df['is_valid'] == 0]['genre'].to_frame() 
    valid_labels = df[df['is_valid'] == 1]['genre'].to_frame()

    train_input_ids, train_attention_masks = get_input_ids_att_masks(train_df)
    valid_input_ids, valid_attention_masks = get_input_ids_att_masks(valid_df)
    print(f'{train_input_ids.shape}, {valid_input_ids.shape}')
    print(f'{train_attention_masks.shape}, {valid_attention_masks.shape}')
    
    print(f'{train_labels.shape}, {valid_labels.shape}') 
    train_dataloader = create_dataloader(train_input_ids, train_attention_masks,
                                        train_labels)
    valid_dataloader = create_dataloader(valid_input_ids, valid_attention_masks,
                                        valid_labels)
    
    epochs = 4

    train(train_dataloader, valid_dataloader, epochs)
    
