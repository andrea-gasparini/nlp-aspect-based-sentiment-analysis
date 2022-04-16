import os

from torchtext.vocab import Vectors, Vocab

def load_pretrained_embeddings(filename: str, cache_dir: str, vocab: Vocab) -> torch.Tensor:
    """
	Loads from a local file static word embedding vectors (e.g. GloVe, FastText)
	and pairs them with the tokens contain in the given vocabulary.

    Args:
            filename: name of the file that contains the vectors
            cache_dir: directory for cached vectors
            vocab: vocabulary of tokens to be embedded
    """
    pretrained_embeddings = Vectors(filename, cache=cache_dir)
    embeddings = torch.randn(len(vocab), pretrained_embeddings.dim)
    initialised = 0

    for i, token in enumerate(vocab.get_itos()):
        token = token.lower()
        if token in pretrained_embeddings.stoi:
            initialised += 1
            embedding = pretrained_embeddings.get_vecs_by_tokens(token)
            embeddings[i] = embedding

    embeddings[vocab["[PAD]"]] = torch.zeros(pretrained_embeddings.dim)

    print(f"initialised {initialised} embeddings")
    print(f"randomly initialised {len(vocabulary) - initialised} embeddings")

    return embeddings


def get_pretrained_model(pretrained_model_name_or_path: str) -> str:
	"""
	Returns the HuggingFace model name or the path to its local directory in case
	the given arg is a valid one.
	"""
	return (pretrained_model_name_or_path
			if os.path.exists(pretrained_model_name_or_path)
			else os.path.basename(os.path.normpath(pretrained_model_name_or_path)))
