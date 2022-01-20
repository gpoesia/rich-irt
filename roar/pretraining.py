#!/usr/bin/env python3

import torch
from torch import nn
import transformers
import pytorch_lightning as pl
from transformers import BertForMaskedLM, BertModel, BertConfig
import random
import wandb
import csv
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm


BOW = 1
EOW = 2
PAD = 3
MASK = 4

def encode(s):
    s = s.encode('ascii', 'ignore').decode()
    return torch.tensor([BOW] + list(map(ord, s)) + [EOW], dtype=torch.long)

def encode_batch(b):
    ts = [encode(s) for s in b]
    max_len = max([t.shape[0] for t in ts])
    for i in range(len(b)):
        ts[i] = torch.cat([ts[i], PAD * torch.zeros(max_len - ts[i].shape[0], dtype=torch.long)])
    return torch.stack(ts)


def random_letter():
    return chr(random.choice(range(ord('a'), ord('z'))))


def transform_insert(s, n_inserts):
    for i in range(n_inserts):
        j = random.randint(0, len(s))
        c = random_letter()
        s = s[:j] + c + s[j:]
    return s


def transform_delete(s, n_deletes):
    n_deletes = min(len(s) - 1, n_deletes)
    kept = sorted(random.sample(range(len(s)), len(s) - n_deletes))
    return ''.join(s[i] for i in kept)


def transform_swap_adj(s, n_swaps):
    if len(s) >= 2:
        for i in range(n_swaps):
            j = random.randint(0, len(s) - 2)
            s = s[:j] + s[j+1] + s[j] + s[j+2:]
    return s


def transform_swap_any(s, n_swaps):
    if len(s) >= 2:
        for i in range(n_swaps):
            j1 = random.randint(0, len(s) - 1)
            j2 = random.randint(0, len(s) - 1)
            j1, j2 = min(j1, j2), max(j1, j2)
            s = s[:j1] + s[j2] + s[j1+1:j2] + s[j1] + s[j2+1:]

    return s

def make_nonwords(words, k=10):
    nonwords = []

    print("Generating pseudowords...")

    for i in tqdm(range(len(words))):
        for _ in range(k):
            w = random.choice(words)

            for i in range(random.randint(1, 8)):
                w = random.choice([
                    lambda w: transform_delete(w, random.randint(0, 2)),
                    lambda w: transform_insert(w, random.randint(0, 2)),
                    lambda w: transform_swap_adj(w, random.randint(0, 2)),
                    lambda w: transform_swap_any(w, random.randint(0, 2)),
                    # Add a suffix from another word.
                    lambda w: random.choice(words)[0:random.randint(0, 5)] + w,
                    # Add a prefix from another word.
                    lambda w: w + random.choice(words)[-random.randint(0, 5):],
                    # Take a prefix.
                    lambda w: w[:random.randint(1, len(w))],
                    # Take a suffix.
                    lambda w: w[random.randint(0, len(w) - 1):],
                ])(w)
            nonwords.append(w)

    nonwords = list(set(nonwords) - set(words))
    return nonwords


class CharBERT(pl.LightningModule):
    def __init__(self):
        bert_config = BertConfig(
            vocab_size=128,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=512,
        )

        super().__init__()
        self.bert = BertForMaskedLM(bert_config)

    def forward(self, batch, labels=None):
        out = self.bert(batch,
                        labels=labels,
                        attention_mask=(batch != PAD).float(),
                        output_hidden_states=True)
        return out.loss, out.hidden_states[-1].mean(dim=1)

    def embed_batch(self, b):
        b = encode_batch(b).to(self.device)
        return self.forward(b)[1]

    def encode_batch(self, batch, mask_prob=None):
        b = encode_batch(batch).to(self.device)

        if mask_prob is None:
            return b, None

        mask = (b != PAD).float() * mask_prob
        mask = torch.bernoulli(mask).long()

        labels = mask * b + (1 - mask) * -100
        b -= (b - PAD) * mask

        return b, labels

    def training_step(self, batch, batch_idx, log_loss='train_loss'):
        enc, labels = self.encode_batch(batch, 0.2)
        loss, _ = self.forward(enc, labels)

        if log_loss is not None:
            self.log(log_loss, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, log_loss='validation_loss')

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, log_loss='test_loss')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    def fit(self):
        with open('/usr/share/dict/words') as f:
            words = list(f)

        words = [w.lower().strip() for w in words]
        random.shuffle(words)

        train_dataloader = torch.utils.data.DataLoader(words[:-1000], shuffle=True, batch_size=256)
        val_dataloader = torch.utils.data.DataLoader(words[-1000:], batch_size=64)

        wandb_logger = WandbLogger('rich-irt')

        trainer = pl.Trainer(gpus=[5], max_epochs=10, logger=wandb_logger)
        trainer.fit(self, train_dataloader, val_dataloader)

        torch.save(self, 'roar_bert.pt')


class CharBERTClassifier(pl.LightningModule):
    def __init__(self):
        bert_config = BertConfig(
            vocab_size=128,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=128,
        )
        self.lr = 5e-5

        super().__init__()
        self.bert = BertModel(bert_config)
        self.output = nn.Linear(bert_config.hidden_size, 1)

    def forward(self, batch):
        bert_out = self.bert(batch) # , attention_mask=(batch != PAD).float())
        embeddings = bert_out.last_hidden_state[:, 0, :]
        y_hat = self.output(embeddings).squeeze(1)
        return y_hat, embeddings

    def embed_batch(self, b):
        b = encode_batch(b).to(self.device)
        return self.forward(b)[1]

    def training_step(self, batch_xy, batch_idx, log_loss='train_loss'):
        batch, y = batch_xy

        enc = encode_batch(batch).to(self.device)
        y_hat, _ = self.forward(enc)
        criterion = torch.nn.BCEWithLogitsLoss()

        loss = criterion(y_hat, torch.tensor(y).type_as(y_hat)).mean()

        if log_loss is not None:
            self.log(log_loss, loss, batch_size=len(batch_xy))

        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        enc = encode_batch(batch).to(self.device)
        y_hat, _ = self.forward(enc)

        accuracy = (y_hat.sigmoid().round() == 
                    torch.tensor(y).type_as(y_hat)).float().mean()
        self.log('validation_accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def fit(self):
        with open('/usr/share/dict/words') as f:
            words = list(f)

        K = 20
        words = 20 * [w.lower().strip() for w in words]
        nonwords = make_nonwords(words, K)

        dataset = [(w, 1) for w in words] + [(w, 0) for w in nonwords]
        random.shuffle(dataset)

        with open('lookup_real_pseudo.csv') as f:
            roar_items = [(r['word'], int(r['realpesudo'] == 'real')) for r in csv.DictReader(f)]

        train_dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=128)
        val_dataloader = torch.utils.data.DataLoader(roar_items, batch_size=64)

        wandb_logger = WandbLogger('rich-irt')

        trainer = pl.Trainer(gpus=[8], max_epochs=20, logger=wandb_logger)

        trainer.fit(self, train_dataloader, val_dataloader)

        torch.save(self, 'roar_bert_supervised_20.pt')


if __name__ == '__main__':
    pass
    #bert = CharBERT()
    bert = CharBERTClassifier()
    bert.fit()
