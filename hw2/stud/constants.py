PAD_TOKEN = "[PAD]"
PAD_INDEX = 0
UNK_TOKEN = "[UNK]"
UNK_INDEX = 1

BERT_OUT_DIM = 768
BERT_PRETRAINED_PATH = f"{MODELS_DIR}bert-base-cased" if os.path.isdir(f"{MODELS_DIR}bert-base-cased") else "bert-base-cased"
