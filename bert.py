import os
import torch
import numpy as np
from datetime import datetime
from dataset.tw_data import TWBertDataModule
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

NEGATIVE = 0
POSITIVE = 1

N_EPOCHS = 2
L_RATE = 2e-5

BATCH_SIZE = 256

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load and create datasets
    data_module = TWBertDataModule("twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt", batch_size=BATCH_SIZE, val_percentage=0.1)
    data_module.setup('fit')
    
    # Training
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=L_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    ts = timestamp('%d-%m-%Y-%H:%M:%S')
    train(model, optimizer, loss_function, data_module.train_dataloader(), data_module.val_dataloader(), device, ts)
    
    path = 'out/models/{}'.format(ts)
    os.makedirs(path, exist_ok = True)
    torch.save(model.state_dict(), os.path.join(path, '{}.pt'.format(ts)))

def train(model, optimizer, loss_function, train_data, val_data, device, ts):
    writer = SummaryWriter('out/logs/{}'.format(ts))

    model.to(device)
    loss_function.to(device)

    for epoch in range(N_EPOCHS):
        train_loss, val_loss = 0, 0
        acc, n_steps, n_data = 0, 0, 0
        val_acc, val_n_steps, val_n_data = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_data):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            prediction = model(ids, mask)

            loss = loss_function(prediction.logits, labels)
            train_loss += loss.item()

            _, big_idx = torch.max(prediction.logits, dim=1)
            acc += accuracy(big_idx, labels)

            n_steps += 1
            n_data += labels.size(0)
            
            if step != 0 and step % 125 == 0:
                loss_step = train_loss / n_steps
                accu_step = (acc * 100) / n_data

                writer.add_scalar('Train / loss', loss_step, (epoch * len(train_data)) + step)
                writer.add_scalar('Train / acc', accu_step, (epoch * len(train_data)) + step)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for step, val_batch in enumerate(val_data):
                val_batch = next(iter(val_data))
                ids = val_batch['input_ids'].to(device)
                mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)

                prediction = model(ids, mask)

                loss = loss_function(prediction.logits, labels)
                val_loss += loss.item()

                _, big_idx = torch.max(prediction.logits, dim=1)
                val_acc += (accuracy(big_idx, labels) * 100) / labels.size(0) 

                val_n_steps += 1
                val_n_data += labels.size(0)

                if step != 0 and step % 125 == 0:
                    loss_step = val_loss / val_n_steps
                    accu_step = (val_acc * 100) / val_n_data

                    writer.add_scalar('Train / loss', loss_step, (epoch * len(val_data)) + step)
                    writer.add_scalar('Train / acc', accu_step, (epoch * len(val_data)) + step)

        epoch_loss = train_loss / n_steps
        epoch_accu = (acc * 100) / n_data
        print(f"Epoch {epoch} Training: Loss = {epoch_loss} Accuracy = {epoch_accu}")

        total_val_loss = val_loss / val_n_steps
        total_val_accu = (val_acc * 100) / val_n_data
        print(f"Epoch {epoch} Validation: Loss = {total_val_loss} Accuracy = {total_val_accu}")

        # writer.add_scalar('Train / loss', train_loss, epoch)
        # writer.add_scalar('Validation / loss', total_val_loss, epoch)
        # writer.add_scalar('Train / acc', epoch_accu, epoch)
        # writer.add_scalar('Validation / acc', total_val_accu, epoch)

    writer.close()

def load_tweets(path: str):
    tweets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tweets.append(line.rstrip())
    return tweets

def split_dataset(tweets, labels):
    np.random.seed(1)

    shuffled_indices = np.random.permutation(len(tweets))
    split = int(0.8 * len(tweets))

    train_indices = shuffled_indices[:split]
    val_indices = shuffled_indices[split:]

    train_tweets = np.array(tweets)[train_indices].tolist()
    train_labels = np.array(labels)[train_indices].tolist()

    val_tweets = np.array(tweets)[val_indices].tolist()
    val_labels = np.array(labels)[val_indices].tolist()

    return (train_tweets, train_labels), (val_tweets, val_labels)

def accuracy(prediction, target):
    return (prediction == target).sum().item()

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

if __name__ == "__main__":
    main()