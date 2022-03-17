import pandas as pd
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

if __name__ == "__main__":

    write_csv_into_txt_files('train_val_data.csv', 'data')
