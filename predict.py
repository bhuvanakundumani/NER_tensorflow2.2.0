import os
import math
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf 
from model import TFNer
from keras.preprocessing.sequence import pad_sequences
from fastprogress.fastprogress import master_bar, progress_bar
from preprocess import split_text_label, padding, createMatrices
from seqeval.metrics import classification_report


def idx_to_label(predictions, correct, idx2Label): 
    
    label_pred = []    
    for sentence in predictions:
        for i in sentence:
            label_pred.append([idx2Label[elem] for elem in i ]) 

    label_correct = []  
    if correct != None:
        for sentence in correct:
            for i in sentence:
                label_correct.append([idx2Label[elem] for elem in i ]) 
        
    return label_correct, label_pred

def predict_single_sentence(sentence, word2Idx, max_seq_len):
        sentence = list(sentence.split(" "))
        sentences = []
        wordIndices = []
        masks = []
        length = len(sentence)

        for word in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:                
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
        maskindices = [1]*len(wordIndices)
        sentences.append(wordIndices)
        masks.append(maskindices)
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=max_seq_len, padding="post")
        masks = tf.keras.preprocessing.sequence.pad_sequences(masks, maxlen=max_seq_len, padding="post")
        return length, masks, padded_inputs

def main():

    logging.basicConfig(format='%(asctime)s - %(levelname)s -  %(message)s', datefmt='%m/%d/%Y ', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data", default=None, type=str, required=True,help="Directory which has the data files for the task")
    parser.add_argument("--model_dir", default=None, type=str, required=True,help="Directory which has the model files for the task")
    parser.add_argument("--predsingle", default=False, type=str, required=False, help="Set it to True and assign the sentence to test_sentence variable.")

    args = parser.parse_args()

    test_batch_size = 64

    # padding sentences and labels to max_length of 128
    max_seq_len = 128
    EMBEDDING_DIM = 100
    test_sentence = "Steve went to Paris"
    idx2Label = pickle.load(open(os.path.join(args.model_dir, "idx2Label.pkl"), 'rb'))
    label2Idx = {v:k for k,v in idx2Label.items()}
    num_labels = len(label2Idx)
    word2Idx = pickle.load(open(os.path.join(args.model_dir, "word2Idx.pkl"), 'rb'))
    embedding_matrix = pickle.load(open(os.path.join(args.model_dir, "embedding.pkl"), 'rb'))
    logger.info("Loaded idx2Label, word2Idx and Embedding matrix pickle files")

    #Loading the model
    testmodel =  TFNer(max_seq_len=max_seq_len, embed_input_dim=len(word2Idx), embed_output_dim=EMBEDDING_DIM, weights=[embedding_matrix], num_labels=num_labels)
    testmodel.load_weights(f"{args.model_dir}/model_weights")
    logger.info("Model weights restored")

    if not args.predsingle:
        #Evaluating on test dataset 
        split_test = split_text_label(os.path.join(args.data, "test.txt"))
        test_sentences, test_labels = createMatrices(split_test, word2Idx, label2Idx)
        test_features, test_labels = padding(test_sentences, test_labels, max_seq_len, padding='post' )
        logger.info(f"Test features shape is {test_features.shape} and labels shape is{test_labels.shape}")
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
        batched_test_dataset = test_dataset.batch(test_batch_size, drop_remainder=True)

        #epoch_bar = master_bar(range(epochs))
        test_pb_max_len = math.ceil(float(len(test_features))/float(test_batch_size))

        true_labels = []
        pred_labels = []

        for sentences_batch, labels_batch in progress_bar(batched_test_dataset, total=test_pb_max_len):
            
            logits = testmodel(sentences_batch)
            temp1 = tf.nn.softmax(logits)   
            preds = tf.argmax(temp1, axis=2)
            true_labels.append(np.asarray(labels_batch))
            pred_labels.append(np.asarray(preds))

        label_correct, label_pred = idx_to_label(pred_labels, true_labels, idx2Label)
        report = classification_report(label_correct, label_pred, digits=4)
        logger.info(f"\nResults for the test dataset") 
        logger.info(f"\n{report}")
   
    else:
        length, masks, padded_inputs = predict_single_sentence(test_sentence, word2Idx, max_seq_len)
        padded_inputs = tf.expand_dims(padded_inputs, 0)
        
        true_labels = None
        pred_labels = []
        pred_logits = []

        for sentence in padded_inputs:
            logits = testmodel(sentence)
            temp1 = tf.nn.softmax(logits) 
            max_values = tf.reduce_max(temp1,axis=-1)

            masked_max_values = max_values * masks 
            preds = tf.argmax(temp1, axis=2)
            pred_labels.append(np.asarray(preds))
            pred_logits.extend(np.asarray(masked_max_values))
        _,label_pred  = idx_to_label(pred_labels, true_labels, idx2Label)
        
        logger.info(f"Results for - \"{test_sentence}\"")
        
        label_pred = label_pred[0][:length] 
        pred_logits = pred_logits[0][:length]
        logger.info(f"Labels predicted are {label_pred}")
        logger.info(f"with a confidence of {pred_logits}")
        

        
if __name__ == "__main__":
    main()