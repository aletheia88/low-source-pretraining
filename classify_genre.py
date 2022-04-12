import torch
import torch.nn as nn
import random
import time
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
        
        # curr outputs[1][-1] shape - torch.Size([4, 1320, 768])
        # outputs shape from tutorial - torch.Size([32, 64, 768])
        
        # (1,768)
        last_hidden_state_cls = outputs[1][-1][:, 0, :]
        print(f'last_hidden_state_cls shape = {last_hidden_state_cls.shape}')
         
        logits = self.classifier(last_hidden_state_cls)
        print(f'logits shape = {logits.shape}')
        
        return logits

def run_pipeline(ds_name):
    
    df = pd.read_csv(ds_name)
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
                                        batch_size=1)

    valid_dataloader = create_dataloader(valid_input_ids, 
                                        valid_attention_masks,
                                        valid_labels,
                                        batch_size=1)

    epochs = 10
    
    if torch.cuda.is_available(): device = torch.device(0)

    avg_train_losses, avg_valid_losses, avg_valid_accuracies = train(
                                                        train_dataloader, 
                                                        valid_dataloader, 
                                                        epochs,
                                                        device)

    return avg_train_losses, avg_valid_losses, avg_valid_accuracies


def label_genres(data_labels):

    labels = []
    label = 1
    
    for i in range(len(data_labels)-1):

        if data_labels[i] == data_labels[i+1]:
            labels.append(label)
        else:
            labels.append(label)
            label += 1
    
    labels.append(label) 
    
    return labels

def train(train_dataloader, valid_dataloader, epochs, device):
 
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
    avg_train_losses = []
    avg_valid_losses = []
    avg_valid_accuracies = []
    
    for epoch_i in range(epochs):
        
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            
            logits = model(b_input_ids, b_attn_mask)
            print(f'logits shape = {logits.shape}')
             
            loss = loss_fn(logits, b_labels)
            print(f'loss = {loss.item()}')
            
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 4 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

            valid_loss, valid_accuracy = evaluate(model, 
                                                valid_dataloader,
                                                torch.device(1))

            print(f'valid loss = {valid_loss}')
            avg_valid_losses.append(valid_loss)
            avg_valid_accuracies.append(valid_accuracy)

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

        print("\n")

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_losses.append(avg_train_loss)

        print("-"*70)

    print("Training complete!")
    print(f'batch_loss: {batch_loss}, total_loss:{total_loss}')
    
    return avg_train_losses, avg_valid_losses, avg_valid_accuracies

def evaluate(model, valid_dataloader, device):

    """ evaluate on validation set """
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    valid_accuracies = []
    valid_losses = []

    for batch in valid_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        valid_losses.append(loss.item())
        
        preds = torch.argmax(logits, dim=1)
        #print(f'preds shape - {preds.shape} labels shape - {b_labels.shape}')
        #print(f'preds - {preds}')
        #print(f'labels - {b_labels}')

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        valid_accuracies.append(accuracy)

    mean_valid_loss = np.mean(valid_losses)
    mean_valid_accuracy = np.mean(val_accuracies)

    return mean_valid_loss, mean_valid_accuracy

def create_dataloader(input_ids, attention_masks, labels, batch_size=4):
    
    ## encode labels ##
    #encoder = OneHotEncoder(handle_unknown='ignore')
    #encoded_labels = pd.DataFrame(encoder.fit_transform(labels).toarray()).values
    #encoded_labels = torch.tensor(encoded_labels)
     
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

    ds_name = 'train_val_data.csv'
    run_pipeline(ds_name)
    
