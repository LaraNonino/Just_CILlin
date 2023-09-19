# Bidirectional CRNN and BERT Embeddings for Sentiment Analysis on Twitter Data
This project aims to perform sentiment analysis on a dataset consisting of tweets. We start by considering two baseline 
results obtained through simple machine learning techniques with different embeddings and compare these to a more 
complex model centered around a state-of-the-art architecture for this kind of task, using BERT embeddings. In the 
comparative study, we evaluate the classification accuracy of each of the models.
The full report can be found at the following [link](https://github.com/LaraNonino/Just_CILlin/blob/011895ccfc1b34bca053bc6c46027ebc7dabcbe6/Report.pdf).
# Setting up
To run the code you first need to install the required packages. A `requirements.txt` file is provided.

To run the exeriments, first save the dataset in a folder called `twitter-datasets`.

# Running different experiments

We provide the different logs of the experiments in the folder called `lightning_logs`. 

To execute the experiments on clean data, first execute the following command and change the training file in the `train_model.py`:

```bash
python src/clean_data.py
```

## Training

To run the training, execute inside the root folder the following command:
```bash
python src/train_model.py --lr=<learning_rate> --n_epochs=<n_epochs> --batch_size=<batch_size> --n_workers=<n_workers> --label_smoothing=<label_smoothing> --sched_step_size=<sched_step_size> --sched_gamma=<sched_gamma>
```

## Predicting

To predict the sentiment of a test file using a model, execute inside the root folder the following command:

```bash
python src/predict_model.py --n_epochs=<n_epochs> --batch_size=<batch_size> --n_workers=<n_workers> --model=<model>
```

## Our model
Our model checkpoint is the final result of the following command:

```bash
python src/train_model.py --n_epochs=2
```

To obtain the predictions with our model, we provide the checkpoint file in the following [link](https://drive.google.com/file/d/1lFxrWlc5EsQ6UJ0oovd07sAGTlZFa_1M/view?usp=sharing). This checkpoint should be saved inside a folder in `lightning_logs/final/checkpoints`. To reproduce the predictions that we provide in the folder `predictions/final`, just execute the commad:

```bash
python src/predict_model.py 
```