import torch
import math
import pandas as pd
from transformers import RobertaForMaskedLM
from tqdm import tqdm
from train_bert_lm import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,\
                                                        SequentialSampler

class genreClassifier(nn.Module):

    def __init__(self, model_path, freeze_bert=False):
        
        super(genreClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 256

        self.bert = RobertaForMaskedLM.from_pretrained(model_path)

        self.classifier = nn.Sequential(
                        nn.Linear(D_in, H),
                        nn.Softmax(),
                        nn.Linear(H, D_out))
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

def train(ds_name):
    return

def get_input_ids_att_masks(data_df):
    
    input_ids = []
    attention_masks = []

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

 
