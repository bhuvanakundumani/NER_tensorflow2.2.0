import ipdb
import tensorflow as tf
from tensorflow.keras import layers

class TFNer(tf.keras.Model):
    def __init__(self, max_seq_len, embed_input_dim, embed_output_dim, num_labels, weights):
        super(TFNer, self).__init__()

        #self.input_words = layers.Input(shape=(max_seq_len,),dtype='int32')
        self.embedding = layers.Embedding(input_dim=embed_input_dim, output_dim=embed_output_dim, weights=weights, input_length=max_seq_len, trainable=False, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dense = layers.Dense(max_seq_len)

    def call(self, inputs):
        #ipdb.set_trace()
        #print("inputs shape is ",inputs.shape)
        x = self.embedding(inputs) # batchsize, max_seq_len, embedding_output_dim
        #print("after embedding", x.shape)
        x = self.bilstm(x) #batchsize, max_seq_len, hidden_dim_bilstm
        #print("after bilstm",x.shape)
        logits = self.dense(x) #batchsize, max_seq_len, max_seq_len
        #print(f"shape of logits", logits.shape)
        return logits


