import pandas as pd
from tqdm import tqdm
import csv

def get_token_freq(f_name):
    
    token_dict = {}
    all_codewords = pd.read_csv(f_name)['codewords'].values

    for codeword in all_codewords:

        tokens = codeword.split(' ')
        
        for token in tokens:

            if token not in token_dict.keys():
                
                token_dict[token] = 1
            
            else:
                token_dict[token] += 1

    return token_dict

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

def get_max_codeword_len(ds_name):

    codewords = pd.read_csv(ds_name)['codewords']
    max_len = 0

    for codeword in codewords:
        if len(codeword.split()) > max_len:
            max_len = len(codeword.split())
    
    return max_len

def convert_txt_to_csv(txt_file, csv_file):
    
    f = open(txt_file)
    csv_f = open(csv_file, 'w')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(['steps', 'training_losses', 'validation_losses'])

    for line in f:
        csv_writer.writerow([float(item) for item in line.split()])
    
    csv_f.close()

if __name__ == "__main__":

    #f_name = 'test_codewords.csv'
    #max_len = get_max_codeword_len(f_name)
    #print(max_len)

    convert_txt_to_csv('lm_records.txt', 'lm_records.csv')
