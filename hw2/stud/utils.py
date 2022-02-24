import os

from transformers import AutoTokenizer


def get_tokenizer(pretrained_model_name_or_path: str) -> AutoTokenizer:
	"""
	Returns an HuggingFace AutoTokenizer, locally loaded if the weights are
	available or downloaded otherwise.
	"""
	if os.path.exists(pretrained_model_name_or_path):
		return AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	else:
		name = os.path.basename(os.path.normpath(pretrained_model_name_or_path))
		return AutoTokenizer.from_pretrained(name)
