from typing import List
from transformers import GPT2Tokenizer


class CharacterLevelTokenizer:
    """
    The class for the character level tokenzier used for the Transformer
    """
    def __init__(self):
        self.start_token = "<SOS>"
        self.end_token = "<EOS>"
        self.vocab = None


    def tokenize(self, text: str) -> List[str]:
        """The function tokenizes the text into characters, starting with <SOS> and ending with the <EOS> token

        Args:
            text (str): The original text to be tokenized

        Returns:
            List[str]: The list with the tokens
        """
        tokens = list(text)  # Split each character
        return [self.start_token] + tokens + [self.end_token]


    def detokenize(self, tokens: List[str]) -> str:
        """The function takes the tokens and reconstructs the original text

        Args:
            tokens (List[str]): The list of tokens

        Returns:
            str: The text after combining the tokens
        """
        reconstructed_str = ''

        for token in tokens:
            if token not in {self.start_token, self.end_token}:
                reconstructed_str += token

        return reconstructed_str

    def set_vocab(self, vocab):
        self.vocab = vocab


class SubwordTokenizer:
    """
       The class for the character subword tokenzier used for the Transformer. I chose GPT2 because the HuggingFace
    documentation for GPT2 was better written.
    """
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        # Add custom tokens for <SOS> and <EOS> if they don't already exist
        if '<SOS>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<SOS>']})
        if '<EOS>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<EOS>']})

        self.start_token_id = self.tokenizer.convert_tokens_to_ids('<SOS>')
        self.end_token_id = self.tokenizer.convert_tokens_to_ids('<EOS>')
        self.end_token = self.end_token_id
        self.vocab = None


    def tokenize(self, text: str) -> List[int]:
        """The function tokenizes using GPT2 pretrained subword tokenizer

        Args:
            text (str): The original text to be tokenized

        Returns:
            List[int]: The list of tokens
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.start_token_id] + tokens + [self.end_token_id]


    def detokenize(self, token_ids: List[int]) -> str:
        """Reconstruct the original text from the tokens

        Args:
            token_ids (List[int]): The list with the tokens

        Returns:
            str: The reconstructed text
        """
        filtered_token_ids = [t for t in token_ids if t not in {self.start_token_id, self.end_token_id}]
        return self.tokenizer.decode(filtered_token_ids, skip_special_tokens=True)

    def set_vocab(self, vocab):
        self.vocab = vocab
