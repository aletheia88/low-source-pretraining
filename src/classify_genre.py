import torch
import torch.nn as nn
import random
import time
import math
import pandas as pd
import numpy as np
import csv
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

        self.bert = RobertaForMaskedLM.from_pretrained('bert-models-upd')
        
        self.classifier = nn.Sequential(
                        nn.Linear(D_in, D_out),
                        nn.Softmax()
                        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        
        last_hidden_state_cls = outputs[1][-1][:, 0, :]
         
        logits = self.classifier(last_hidden_state_cls)
        
        return logits

def evaluate_model(model, test_ds_name, device, batch_size):

    df_t = pd.read_csv(test_ds_name)
    test_df = df_t['codewords']
    test_genres = df_t['genre'].values
    
    test_labels = torch.tensor(label_genres(test_genres))
    test_input_ids, test_attention_masks = get_input_ids_att_masks(test_df)
    
    test_dataloader = create_dataloader(test_input_ids,
                                        test_attention_masks,
                                        test_labels,
                                        batch_size)
 

    test_loss, test_accuracy = evaluate(model, test_dataloader, device)
    
    return {'test_loss':test_loss,
            'test_accuracy':test_accuracy}

def get_trained_model(init_model, train_val_ds_name, csv_file, batch_size, epochs, device):

    df = pd.read_csv(train_val_ds_name)

    train_df = df[df['is_valid'] == 0]['codewords'] 
    valid_df = df[df['is_valid'] == 1]['codewords']

    train_genres = df[df['is_valid'] == 0]['genre'].values
    valid_genres = df[df['is_valid'] == 1]['genre'].values

    train_labels = torch.tensor(label_genres(train_genres))
    valid_labels = torch.tensor(label_genres(valid_genres))

    train_input_ids, train_attention_masks = get_input_ids_att_masks(train_df)
    valid_input_ids, valid_attention_masks = get_input_ids_att_masks(valid_df)

    train_dataloader = create_dataloader(train_input_ids, 
                                        train_attention_masks,
                                        train_labels,
                                        batch_size)

    valid_dataloader = create_dataloader(valid_input_ids, 
                                        valid_attention_masks,
                                        valid_labels,
                                        batch_size)
   
    #if torch.cuda.is_available(): device = torch.device("cuda:0")

    trained_model = train(
                init_model,
                train_dataloader,
                valid_dataloader,
                epochs,
                device,
                csv_file)
    
    return trained_model

def label_genres(data_labels):

    labels = []
    label = 0
    
    for i in range(len(data_labels)-1):

        if data_labels[i] == data_labels[i+1]:
            labels.append(label)
        else:
            labels.append(label)
            label += 1
    
    labels.append(label) 
    
    return labels

def train(model, train_dataloader, valid_dataloader, epochs, device, csv_fname):
 
    loss_fn = nn.CrossEntropyLoss()
    
    #model = genreClassifier(freeze_bert=False)
    #model.to(device)

    optimizer = AdamW(model.parameters(),
                        lr=5e-5,
                        eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    '''
    csv_file = open(csv_fname, 'w') 
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['train_losses', 'valid_losses', 'valid_accuracies'])
    '''

    record_df = pd.DataFrame(columns=('train_losses', 'valid_losses', 'valid_accuracies'))
        
    for epoch_i in range(epochs):
        
        #print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        #print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 4 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                #print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        
        avg_train_loss = total_loss / len(train_dataloader)
        #print("-"*70)
                
        valid_loss, valid_accuracy = evaluate(model,
                                            valid_dataloader,
                                            device)

        #csv_writer.writerow([avg_train_loss, valid_loss, valid_accuracy])
        record_df.loc[epoch_i] = [avg_train_loss, valid_loss, valid_accuracy]
        record_df.to_csv(csv_fname)

        time_elapsed = time.time() - t0_epoch
        #print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {valid_loss:^10.6f} | {valid_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        #print("-"*70)
        #print("\n")
    
    #csv_file.close()
    print("Training complete!")

    return model

def evaluate(model, eval_dataloader, device):

    """ evaluate on validation/test set """
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    eval_accuracies = []
    eval_losses = []

    for batch in eval_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        eval_losses.append(loss.item())
        
        preds = torch.argmax(logits, dim=1)

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        eval_accuracies.append(accuracy)

    return np.mean(eval_losses), np.mean(eval_accuracies)

def create_dataloader(input_ids, attention_masks, labels, batch_size=4):
    
    data = TensorDataset(input_ids, attention_masks, labels)
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

    train_val_ds_name = 'train_val_codewords.csv'
    test_ds_name = 'test_codewords.csv'
    csv_file = 'training_records.csv'

