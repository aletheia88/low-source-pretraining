### Exploring Low-Resource Language Model Pretraining for Musical Information Retrieval

**Current method for classification tasks**
* Training a language model on unlabeled data that learns to predict the next word or masked words in a sequence
* Adding a classification layer and train on a small set of labeled data; treating language model as a fixed feature extractor

**Problems and our ideas**
People typically just use a pretrained model from some other task. State-of-the-art models nowadays are gigantic, mostly trained by companies with access to enormous compute power. However, universities and colleges don’t have the resources to train these models. Research labs don’t have the resources to train giant LMs, but they do have the computing resources to train non-trivial models.

*We would like to explore the idea of moving FROM*

Giant LM (fixed) → linear classification layer (trained) TO
Giant LM (fixed) → small LM (trained) → linear classification layer (trained)

We want to know how much LM pretraining helps when you have a relatively small amount of data.

**Plans**

* Get training data
* Extract MFCCs from the training data (using default librosa settings)
* Do k-means clustering to learn a set of K=256 clusters on the MFCC features
* Use the learned K=256 clusters to quantize each MFCC feature vector into a codeword
* Train BERT LM on codeword sequences
* Finetune BERT classifier on training labels


