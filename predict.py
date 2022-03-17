import csv
import pandas as pd
import numpy as np
import argparse
from fastai.text.all import *
from fastai.text import *
from fastai import *

def predict(file_name, model):

    """
    Use trained model to predict musical genres of each audio clip in test set given its codewords
    Return overall accuracy of prediction
    """

    test_df = pd.read_csv(file_name)
    learner = load_learner(model)
    num_accu = 0

    for i in range(len(test_df)):
        
        pred = learner.predict(test_df['codewords'][i])[0]
        
        if test_df['genre'][i] == pred:
            num_accu += 1 

    return num_accu/len(test_df)

def main(args):
    
    accuracy = predict(args.file, args.model)
    print(f'accuracy is {accuracy*100}')
    return accuracy

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', required=True,
                        help='read in test csv file')
    parser.add_argument('--model', '-m', required=True,
                        help='takes in model for testing')
    args = parser.parse_args()
    main(args)

