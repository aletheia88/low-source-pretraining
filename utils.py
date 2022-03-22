import pandas as pd
from tqdm import tqdm
import csv

def write_csv_into_txt_files(csv_path, txt_path):
    
    text_csv = pd.read_csv(csv_path)

    for i, row in text_csv.iterrows():

        txt_file = open(f"{txt_path}/{row['audio_id'][:-4]}.txt", "w")
        txt_file.writelines(row['codewords'])
        txt_file.close()

def format(codewords):
    
    tokens = codewords.split(' ')
      
    for i, s in enumerate(tokens):
        tokens[i] = f'<s>{s[1:]}</s>'
    
    return tokens


def one_hot_encode(data, col_name):
    
    one_hot_encoded_labels = []
    
    for i in tqdm(range(len(data))):
        
        label = pd.get_dummies(data, columns=[col_name]).iloc[i][4:].values
        one_hot_encoded_labels.append(list(label))

    return one_hot_encoded_labels


