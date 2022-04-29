#!/usr/bin/env python
# coding: utf-8

import os
import wandb
import torch
import torchtext
import pytorch_lightning as pl

from evaluate import read_dataset

from nltk import TreebankWordTokenizer
from pytorch_lightning.loggers import WandbLogger

from stud.dataset import ABSADataModule
from stud.pl_models import PlAspectClassifier
from stud import utils, constants as const


ROOT_DIR = '../'
DATA_DIR = f'{ROOT_DIR}data/'
MODELS_DIR = f'{ROOT_DIR}model/'
EMBEDDINGS_DIR = f'{MODELS_DIR}embeddings/'

assert os.path.isdir(ROOT_DIR), f"{ROOT_DIR} is not a valid directory"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pre-trained static embeddings
GLOVE_DIR = f"{EMBEDDINGS_DIR}glove/"
GLOVE_FILENAME = "glove.6B.300d.txt"

# Laptop datasets
LAPTOP_TRAIN_JSON = os.path.join(DATA_DIR, "laptops_train.json")
LAPTOP_DEV_JSON = os.path.join(DATA_DIR, "laptops_dev.json")
assert os.path.isfile(LAPTOP_TRAIN_JSON), f"{LAPTOP_TRAIN_JSON} does not contain a valid train dataset"
assert os.path.isfile(LAPTOP_DEV_JSON), f"{LAPTOP_DEV_JSON} does not contain a valid development dataset"

# Restaurant datasets
RESTAURANT_TRAIN_JSON = os.path.join(DATA_DIR, "restaurants_train.json")
RESTAURANT_DEV_JSON = os.path.join(DATA_DIR, "restaurants_dev.json")
assert os.path.isfile(RESTAURANT_TRAIN_JSON), f"{RESTAURANT_TRAIN_JSON} does not contain a valid train dataset"
assert os.path.isfile(RESTAURANT_DEV_JSON), f"{RESTAURANT_DEV_JSON} does not contain a valid development dataset"

SEED = 42

pl.seed_everything(SEED)
torch.backends.cudnn.deterministic = True  # will use only deterministic algorithms

WANDB_PROJECT_AB = 'nlp_hw2-AB'
WANDB_PROJECT_CD = 'nlp_hw2-CD'
WANDB_PROJECT = WANDB_PROJECT_AB
WANDB_KEY = ""
wandb.login(key=WANDB_KEY)

BATCH_SIZE = 8

utils.nltk_downloads()

TRAIN_DATASET = (read_dataset(LAPTOP_TRAIN_JSON) + read_dataset(RESTAURANT_TRAIN_JSON)
                 if WANDB_PROJECT != WANDB_PROJECT_CD
                 else read_dataset(RESTAURANT_TRAIN_JSON))

DEV_DATASET = (read_dataset(LAPTOP_DEV_JSON) + read_dataset(RESTAURANT_DEV_JSON)
               if WANDB_PROJECT != WANDB_PROJECT_CD
               else read_dataset(RESTAURANT_DEV_JSON))

data_module = ABSADataModule(TRAIN_DATASET,
                             DEV_DATASET,
                             tokenizer=TreebankWordTokenizer(),
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             has_category=WANDB_PROJECT == WANDB_PROJECT_CD)
data_module.setup()

if not os.path.isfile(f"{GLOVE_DIR}{GLOVE_FILENAME}"):
    torchtext.vocab.GloVe(name="6B", dim=300, cache=GLOVE_DIR)

glove_embeddings = utils.load_pretrained_embeddings(GLOVE_FILENAME,
                                                    GLOVE_DIR,
                                                    data_module.vocabs["text"])

MODE = "ab" if WANDB_PROJECT != WANDB_PROJECT_CD else "cd"
BERT_PRETRAINED_PATH = (f"{MODELS_DIR}bert-base-cased"
                        if os.path.isdir(f"{MODELS_DIR}bert-base-cased")
                        else "bert-base-cased")

POS = False
BERT = True
BERT_FINETUNE = False
ATTENTION = False
ATTENTION_SIMPLE = True
ATTENTION_CONCAT = False

label_key = utils.get_label_key(MODE)

NUM_CLASSES = (len(data_module.vocabs[label_key])
               if MODE != "cd"
               else len(data_module.vocabs["categories"]) * (len(data_module.vocabs["category_polarities"]) + 1))

hparams = {"vocab_size": len(data_module.vocabs["text"]),
           "hidden_dim": 128,
           "embedding_dim": glove_embeddings.shape[1],
           "pos_embedding": POS,
           "pos_embedding_dim": 120,
           "pos_vocab_size": len(data_module.vocabs["pos"]),
           "bert_embedding": BERT,
           "bert_finetuning": BERT_FINETUNE,
           "bert_layers_to_merge": [-1, -2, -3, -4],
           "bert_layer_pooling_strategy": "mean" if MODE == "cd" else "second_to_last",
           "bert_wordpiece_pooling_strategy": "mean",
           "bert_model_name_or_path": BERT_PRETRAINED_PATH,
           "pack_lstm_input": True,
           "label_vocab": data_module.vocabs[label_key if MODE != "cd" else "categories"],
           "polarity_vocab": data_module.vocabs["category_polarities"] if MODE == "cd" else None,
           "num_classes": NUM_CLASSES,
           "bidirectional": False,
           "num_layers": 1,
           "dropout": 0.5,
           "max_epochs": 150,
           "attention": ATTENTION,
           "attention_heads": 12,
           "attention_dropout": 0.2,
           "attention_concat": ATTENTION_CONCAT,
           "attention_simple": ATTENTION_SIMPLE,
           "mode": MODE,
           "batch_size": BATCH_SIZE}

if hparams["bidirectional"]:
    MODEL_NAME = "BiLSTM"
else:
    MODEL_NAME = "LSTM"

MODEL_NAME += " + GloVe"
if hparams["bert_embedding"]:
    MODEL_NAME += f" + BERT_{hparams['bert_layer_pooling_strategy']}"
    if hparams["bert_finetuning"]: MODEL_NAME += " finetuned"
if hparams["pos_embedding"]: MODEL_NAME += " + POS"
if hparams["attention"]:
    MODEL_NAME += " + attention"
    if not hparams["attention_simple"]: MODEL_NAME += " transformer"
    if hparams["attention_concat"]: MODEL_NAME += " + concat"

MODEL_NAME = f"{MODE}_{MODEL_NAME}"

print(MODEL_NAME)

early_stopping = pl.callbacks.EarlyStopping(monitor='valid_aspect_polarity_classification_f1',
                                            patience=10,
                                            verbose=True,
                                            mode='max')

check_point_callback = pl.callbacks.ModelCheckpoint(monitor='valid_aspect_polarity_classification_f1',
                                                    verbose=True,
                                                    save_top_k=2,
                                                    save_last=False,
                                                    mode='max',
                                                    dirpath=MODELS_DIR,
                                                    filename=MODEL_NAME + '-{epoch}-{valid_loss:.4f}-'
                                                                          '{valid_aspect_identification_f1:.3f}-'
                                                                          '{valid_aspect_polarity_classification_f1:.3f}')

wandb_logger = WandbLogger(offline=False, project=WANDB_PROJECT, name=MODEL_NAME)

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    logger=wandb_logger,
    val_check_interval=1.0,
    max_epochs=hparams["max_epochs"],
    callbacks=[early_stopping, check_point_callback]
)

model = PlAspectClassifier(
    hparams,
    embeddings=glove_embeddings,
    ignore_index=const.PAD_INDEX if MODE != "cd" else -100
)
trainer.fit(model, datamodule=data_module)

wandb.finish()
