import os
import math
import pickle
import logging
import argparse
import itertools
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

def main():

    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s', datefmt='%m/%d/%Y ', level=logging.INFO)
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data", default=None, type=str, required=True,help="Directory which has the data files for the task")
    parser.add_argument("--output", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")


    args = parser.parse_args()

    if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
   

    train_batch_size = 32
    valid_batch_size = 64
    test_batch_size = 64

    # padding sentences and labels to max_length of 128
    max_seq_len = 128
    EMBEDDING_DIM = 100
    epochs = 10

    split_train = split_text_label(os.path.join(args.data, "train.txt"))
    split_valid = split_text_label(os.path.join(args.data, "valid.txt"))
    split_test = split_text_label(os.path.join(args.data, "test.txt"))

    labelSet = set()
    wordSet = set()
    # words and labels 
    for data in [split_train, split_valid, split_test]:
        for labeled_text in data:
            for word, label in labeled_text:
                labelSet.add(label)
                wordSet.add(word.lower())

    # Sort the set to ensure '0' is assigned to 0
    sorted_labels = sorted(list(labelSet), key=len)

    # Create mapping for labels
    label2Idx = {}
    for label in sorted_labels:
        label2Idx[label] = len(label2Idx)

    num_labels = len(label2Idx)
    idx2Label = {v: k for k, v in label2Idx.items()}
    
    pickle.dump(idx2Label,open(os.path.join(args.output, "idx2Label.pkl"), 'wb'))
    logger.info("Saved idx2Label pickle file")

    # Create mapping for words 
    word2Idx = {}
    if len(word2Idx) == 0:
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
    for word in wordSet:
        word2Idx[word] = len(word2Idx)
    logger.info("Total number of words is : %d ", len(word2Idx))

    pickle.dump(word2Idx, open(os.path.join(args.output, "word2Idx.pkl"), 'wb'))
    logger.info("Saved word2Idx pickle file")

    # Loading glove embeddings 
    embeddings_index = {}
    f = open('embeddings/glove.6B.100d.txt', encoding="utf-8")
    for line in f:
        values = line.strip().split(' ')
        word = values[0] # the first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') #100d vectors representing the word
        embeddings_index[word] = coefs
    f.close()
    logger.info("Glove data loaded")

    #print(str(dict(itertools.islice(embeddings_index.items(), 2))))

    embedding_matrix = np.zeros((len(word2Idx), EMBEDDING_DIM))
    
    # Word embeddings for the tokens    
    for word,i in word2Idx.items():
        embedding_vector = embeddings_index.get(word)      
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    pickle.dump(embedding_matrix, open(os.path.join(args.output, "embedding.pkl"), 'wb'))
    logger.info("Saved Embedding matrix pickle")
  
    # Interesting - to check how many words were not there in Glove Embedding
    # indices = np.where(np.all(np.isclose(embedding_matrix, 0), axis=1))
    # print(len(indices[0]))

    train_sentences, train_labels = createMatrices(split_train, word2Idx, label2Idx)
    valid_sentences, valid_labels = createMatrices(split_valid, word2Idx, label2Idx)
    test_sentences, test_labels = createMatrices(split_test, word2Idx, label2Idx)
    
    train_features, train_labels = padding(train_sentences, train_labels, max_seq_len, padding='post' )
    valid_features, valid_labels = padding(valid_sentences, valid_labels, max_seq_len, padding='post' )
    test_features, test_labels = padding(test_sentences, test_labels, max_seq_len, padding='post' )

    logger.info(f"Train features shape is {train_features.shape} and labels shape is{train_labels.shape}")
    logger.info(f"Valid features shape is {valid_features.shape} and labels shape is{valid_labels.shape}")
    logger.info(f"Test features shape is {test_features.shape} and labels shape is{test_labels.shape}")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))


    shuffled_train_dataset = train_dataset.shuffle(buffer_size=train_features.shape[0], reshuffle_each_iteration=True)

    batched_train_dataset = shuffled_train_dataset.batch(train_batch_size, drop_remainder=True)
    batched_valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=True)
    batched_test_dataset = test_dataset.batch(test_batch_size, drop_remainder=True)

    epoch_bar = master_bar(range(epochs))
    train_pb_max_len = math.ceil(float(len(train_features))/float(train_batch_size))
    valid_pb_max_len = math.ceil(float(len(valid_features))/float(valid_batch_size))
    test_pb_max_len = math.ceil(float(len(test_features))/float(test_batch_size))

    model = TFNer(max_seq_len=max_seq_len, embed_input_dim=len(word2Idx), embed_output_dim=EMBEDDING_DIM, weights=[embedding_matrix], num_labels=num_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_log_dir = f"{args.output}/logs/train"
    valid_log_dir = f"{args.output}/logs/valid"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    
    train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    valid_loss_metric = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)

    def train_step_fn(sentences_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits = model(sentences_batch) # batchsize, max_seq_len, num_labels
            loss = scce(labels_batch, logits) #batchsize,max_seq_len
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        return loss, logits

    def valid_step_fn(sentences_batch, labels_batch):
        logits = model(sentences_batch)
        loss = scce(labels_batch, logits)
        return loss, logits
    
    for epoch in epoch_bar:
        with train_summary_writer.as_default():
            for sentences_batch, labels_batch in progress_bar(batched_train_dataset, total=train_pb_max_len, parent=epoch_bar) :
                
                loss, logits = train_step_fn(sentences_batch, labels_batch)
                train_loss_metric(loss)
                epoch_bar.child.comment = f'training loss : {train_loss_metric.result()}'
            tf.summary.scalar('training loss', train_loss_metric.result(), step=epoch)
            train_loss_metric.reset_states()

        with valid_summary_writer.as_default():
            for sentences_batch, labels_batch in progress_bar(batched_valid_dataset, total=valid_pb_max_len, parent=epoch_bar):
                loss, logits = valid_step_fn(sentences_batch, labels_batch)
                valid_loss_metric.update_state(loss)
                
                epoch_bar.child.comment = f'validation loss : {valid_loss_metric.result()}'
          
            # Logging after each Epoch !
            tf.summary.scalar('valid loss', valid_loss_metric.result(), step=epoch)
            valid_loss_metric.reset_states()

    model.save_weights(f"{args.output}/model_weights",save_format='tf')  
    logger.info(f"Model weights saved")
   
   
    #Evaluating on test dataset 

    test_model =  TFNer(max_seq_len=max_seq_len, embed_input_dim=len(word2Idx), embed_output_dim=EMBEDDING_DIM, weights=[embedding_matrix], num_labels=num_labels)
    test_model.load_weights(f"{args.output}/model_weights")
    logger.info(f"Model weights restored")

    true_labels = []
    pred_labels = []

    for sentences_batch, labels_batch in progress_bar(batched_test_dataset, total=test_pb_max_len):
        
        logits = test_model(sentences_batch)
        temp1 = tf.nn.softmax(logits)       
        preds = tf.argmax(temp1, axis=2)
        true_labels.append(np.asarray(labels_batch))
        pred_labels.append(np.asarray(preds))

    label_correct, label_pred = idx_to_label(pred_labels, true_labels, idx2Label)
    report = classification_report(label_correct, label_pred, digits=4)
    logger.info(f"Results for the test dataset")
    logger.info(f"\n{report}")

if __name__ == "__main__":
    main()
   
    