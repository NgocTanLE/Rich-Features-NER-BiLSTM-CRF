# Rich-Features-NER-BiLSTM-CRF
Build a Neural Network Model based on Rich Features for Named Entity Recognition

## Datasets Files format:
Word Lemma Brown_Cluster POS CHUNK NE_label

## Baseline:
python factored_bi_lstm.py -d eng.train.added -t eng.testa.added

## Train a new Model with all features:
python factored_bi_lstm_5fea.py -d eng.train.added -t eng.testa.added
