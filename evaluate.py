import torch
from decepticon import DecoderOnlyTransformer
import argparse
from tokenizers_classes import CharacterLevelTokenizer, SubwordTokenizer
import os
import json
from utils import generate_sequence


def main(d_model: int, num_heads: int, d_ff: int, num_layers: int, tokenizer: object, model_name: str):
    with open(os.path.join('vocab', model_name + '.txt')) as file:
        vocab_size = int(file.readline().strip())
        pad_token_idx = int(file.readline().strip())
        vocab = json.load(file)
        tokenizer.set_vocab(vocab)

    model = DecoderOnlyTransformer(vocab_size=vocab_size, pad_token_idx=pad_token_idx, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers)
    model.load_state_dict(torch.load(os.path.join('models', model_name + '.pth')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(generate_sequence(model, device, tokenizer, "Richard:", vocab_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the training with different configs for the model")
    parser.add_argument('--size', choices=['small', 'large'], required=True,
                        help="Specify the size of the model: 'small' or 'large'.")
    parser.add_argument('--tokenizer', choices=['char', 'word'], required=True,
                        help="Specify the tokenizer: 'char' or 'word'.")

    args = parser.parse_args()

    if args.size == 'small' and args.tokenizer == 'char':
        main(d_model=384, num_heads=8, d_ff=1536, num_layers=6, tokenizer=CharacterLevelTokenizer(), model_name='small_char')
    elif args.size == 'small' and args.tokenizer == 'word':
        main(d_model=384, num_heads=8, d_ff=1536, num_layers=6, tokenizer=SubwordTokenizer(), model_name='small_word')