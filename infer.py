import os
import math
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf 
from model import TFNer
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from fastprogress.fastprogress import master_bar, progress_bar
from preprocess import split_text_label, padding, createMatrices
from seqeval.metrics import classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s -  %(message)s', datefmt='%m/%d/%Y ', level=logging.INFO)
logger = logging.getLogger(__name__)
class Ner:
    def __init__(self ,model_dir: str):
        
        self.idx2Label = pickle.load(open(os.path.join(model_dir, "idx2Label.pkl"), 'rb'))
        self.label2Idx = {v:k for k,v in self.idx2Label.items()}
        self.word2Idx = pickle.load(open(os.path.join(model_dir, "word2Idx.pkl"), 'rb'))
        self.embedding_matrix = pickle.load(open(os.path.join(model_dir, "embedding.pkl"), 'rb'))
        self.test_batch_size = 64
        self.max_seq_len = 128
        self.EMBEDDING_DIM = 100
        self.num_labels = len(self.label2Idx)
        
        self.model = self.load_model(model_dir)
        

    def load_model(self, model_dir):
       
        model =  TFNer(max_seq_len=self.max_seq_len, embed_input_dim=len(self.word2Idx), embed_output_dim=self.EMBEDDING_DIM, weights=[self.embedding_matrix], num_labels=self.num_labels)
        model.load_weights(f"{model_dir}/model_weights")
        #tf.train.Checkpoint.restore(f"{model_dir}/model_weights")
        logger.info("Model weights restored")
        return model
    
    def preprocess(self, text):
        sentence = list(text.split(" "))
        sentences = []
        wordIndices = []
        masks = []
        length = len(sentence)

        for word in sentence:
            if word in self.word2Idx:
                wordIdx = self.word2Idx[word]
            elif word.lower() in self.word2Idx:
                wordIdx = self.word2Idx[word.lower()]                 
            else:                
                wordIdx = self.word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
        maskindices = [1]*len(wordIndices)
        sentences.append(wordIndices)
        masks.append(maskindices)
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=self.max_seq_len, padding="post")
        masks = tf.keras.preprocessing.sequence.pad_sequences(masks, maxlen=self.max_seq_len, padding="post")
        return length, masks, padded_inputs

    def idx_to_label(self, predictions, correct): 
    
        label_pred = []    
        for sentence in predictions:
            for i in sentence:
                label_pred.append([self.idx2Label[elem] for elem in i ]) 

        label_correct = []  
        if correct != None:
            for sentence in correct:
                for i in sentence:
                    label_correct.append([self.idx2Label[elem] for elem in i ]) 
            
        return label_correct, label_pred


    def predict(self, text):  
        
        length, masks, padded_inputs = self.preprocess(text)
        padded_inputs = tf.expand_dims(padded_inputs, 0)
        
        true_labels = None
        pred_labels = []
        pred_logits = []
        for sentence in padded_inputs:
            logits = self.model(sentence)
            temp1 = tf.nn.softmax(logits) 
            max_values = tf.reduce_max(temp1,axis=-1)
            masked_max_values = max_values * masks 
            preds = tf.argmax(temp1, axis=2)
            pred_labels.append(np.asarray(preds))
            pred_logits.extend(np.asarray(masked_max_values))
        _,label_pred  = self.idx_to_label(pred_labels, true_labels)
        label_pred = label_pred[0][:length] 
        pred_logits = pred_logits[0][:length]
        words = word_tokenize(text)
        assert len(label_pred) == len(words)
        zip_val = zip(words, label_pred, pred_logits)
        
        output = [{"word":word,"tag":label,"confidence":confidence} for  word, label, confidence in zip_val]

        logger.info(f"Labels predicted are {label_pred}")
        logger.info(f"with a confidence of {pred_logits}")
        return output


        
if __name__ == "__main__":
    text = "Steve went to Paris"
    model_dir = "model_output"
    Nermodel = Ner(model_dir)
    output = Nermodel.predict(text)
