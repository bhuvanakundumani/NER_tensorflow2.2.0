from keras.preprocessing.sequence import pad_sequences

def split_text_label(filename):
    '''
    Reads a file named filename, extracts the text and the labels and stores
    them in an array.
     
    returns [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ] 

    '''
    f = open(filename)
    split_labeled_text = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                split_labeled_text.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1].rstrip("\n")])
    if len(sentence) > 0:
        split_labeled_text.append(sentence)
        sentence = []
    return split_labeled_text

def padding(sentences, labels, max_len, padding='post'):
    padded_sentences = pad_sequences(sentences, max_len, padding='post')
    padded_labels = pad_sequences(labels, max_len, padding='post')
    return padded_sentences, padded_labels

def createMatrices(data, word2Idx, label2Idx):
    sentences = []
    labels = []
    for split_labeled_text in data:
        wordIndices = []
        labelIndices = []
        for word, label in split_labeled_text:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:                
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])
    
        sentences.append(wordIndices)
        labels.append(labelIndices)
    return sentences, labels

