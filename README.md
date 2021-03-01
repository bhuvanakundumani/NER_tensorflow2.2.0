
NER TF 2.0 implementation for CoNLL-2003 NER dataset

A detailed blog for this implementation is available at

https://medium.com/analytics-vidhya/ner-tensorflow-2-2-0-9f10dcf5a0a

'data' folder has the dataset. Model output files will be written in the 'model_ouput; directory. 

To train the model
```bash
python3 train.py --data data --output  model_output --overwrite True 
```

Predict on test Dataset
```bash
python3 predict.py --data data --model_dir model_output
```
Predict a single sentence - Assign the sentence to test_sentence variable. 

```bash
python3 predict.py --data data --model_dir model_output --predsingle True
```

Visualisations 
```bash
tensorboard --logdir=model_output/logs/train --port=6006 --bind_all
tensorboard --logdir=model_output/logs/valid --port=6006 --bind_all
```

