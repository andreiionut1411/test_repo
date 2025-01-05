from decepticon import causal_language_model_loss, DecoderOnlyTransformer
from dataset_processor import ShakespeareDatasetProcessor
from tokenizers_classes import CharacterLevelTokenizer, SubwordTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils import evaluate_bleu_and_rouge
import argparse
import os
import json


class TextDataset(Dataset):
    def __init__(self, tokenized_samples, vocab, pad_token_idx, tokenizer):
        self.tokenized_samples = tokenized_samples
        self.vocab = vocab
        self.pad_token_idx = pad_token_idx
        self.vocab_size = len(self.vocab)
        self.tokenizer_type = type(tokenizer)

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        if self.tokenizer_type == CharacterLevelTokenizer:
            token_indices = torch.tensor([self.vocab[token] for token in self.tokenized_samples[idx]])
        else:
            token_indices = torch.tensor(self.tokenized_samples[idx])

        return token_indices

def collate_fn(batch, pad_token_idx):
    inputs_padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)
    # Shift targets to the right
    targets_padded = inputs_padded.clone()
    targets_padded[:, :-1] = inputs_padded[:, 1:]
    targets_padded[:, -1] = pad_token_idx  # The last target token is padding
    return inputs_padded, targets_padded


def main(d_model: int, num_heads: int, d_ff: int, num_layers: int, tokenizer: object, model_name: str):
    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists('losses'):
        os.mkdir('losses')

    if not os.path.exists('vocab'):
        os.mkdir('vocab')

    processor = ShakespeareDatasetProcessor('Shakespeare_data.csv')
    processor.load_data()
    samples = processor.process_data()
    samples = [' '.join(sample) for sample in samples]
    train_losses = []
    test_losses = []
    perplexity_values = []

    processor.split_data(samples)
    train_tokens = [tokenizer.tokenize(sample) for sample in processor.train_samples]
    dev_tokens = [tokenizer.tokenize(sample) for sample in processor.dev_samples]
    eval_tokens = [tokenizer.tokenize(sample) for sample in processor.eval_samples]

    if type(tokenizer) == CharacterLevelTokenizer:
        all_tokens = [char for tokens in (train_tokens + dev_tokens + eval_tokens) for char in tokens]
        vocab = {char: idx for idx, char in enumerate(set(all_tokens))}
        vocab['<PAD>'] = len(vocab)
        pad_token_idx = vocab['<PAD>']
        tokenizer.set_vocab(vocab)
    elif type(tokenizer) == SubwordTokenizer:
        if '<PAD>' not in tokenizer.tokenizer.get_vocab():
            tokenizer.tokenizer.add_special_tokens({'additional_special_tokens': ['<PAD>']})
        pad_token_idx = tokenizer.tokenizer.convert_tokens_to_ids('<PAD>')
        vocab = tokenizer.tokenizer.get_vocab()
        tokenizer.set_vocab(vocab)

    # Create the data loaders
    train_dataset = TextDataset(train_tokens, vocab, pad_token_idx, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    dev_dataset = TextDataset(dev_tokens, vocab, pad_token_idx, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=50, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    eval_dataset = TextDataset(eval_tokens, vocab, pad_token_idx, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=50, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))

    vocab_size = len(train_dataset.vocab)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, pad_token_idx=train_dataset.pad_token_idx, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers)

    # We save the necessary details about the vocabulary, so that we can get them back when we want to generate sequences
    # after the model was trained.
    with open(os.path.join('vocab', model_name + '.txt'), 'w') as file:
        file.write(str(vocab_size) + '\n')
        file.write(str(pad_token_idx) + '\n')
        json.dump(vocab, file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    num_epochs = 2

    # Define scheduler
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warm-up
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            loss = causal_language_model_loss(logits, targets, train_dataset.pad_token_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.3f}")

        # Online Perplexity Calculation on the hold-out set
        model.eval()
        total_log_likelihood = 0.0
        total_token_count = 0
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Dev PPL"):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass through the model
                logits = model(inputs)

                # Compute the causal loss (same as during training)
                loss = causal_language_model_loss(logits, targets, pad_token_idx=train_dataset.pad_token_idx)

                total_loss += loss.item()
                non_pad_tokens = (targets != train_dataset.pad_token_idx).sum().item()
                total_log_likelihood += loss.item() * non_pad_tokens  # Multiply by batch size
                total_token_count += non_pad_tokens

        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_token_count
        perplexity = torch.exp(torch.tensor(avg_log_likelihood))
        avg_dev_loss = total_loss / len(dev_dataloader)
        test_losses.append(avg_dev_loss)
        perplexity_values.append(perplexity.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] - Dev Loss: {avg_dev_loss:.3f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Dev Perplexity: {perplexity.item():.3f}")


    # Evaluate the model
    model.eval()
    total_log_likelihood = 0.0
    total_token_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Test PPL"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            # Compute causal language model loss (similar to training)
            loss = causal_language_model_loss(logits, targets, pad_token_idx=train_dataset.pad_token_idx)

            non_pad_tokens = (targets != train_dataset.pad_token_idx).sum().item()
            total_log_likelihood += loss.item() * non_pad_tokens
            total_token_count += non_pad_tokens

    # Calculate final test perplexity
    avg_log_likelihood = total_log_likelihood / total_token_count
    test_perplexity = torch.exp(torch.tensor(avg_log_likelihood))
    print(f"Test Perplexity: {test_perplexity.item():.3f}")

    # I save the losses into a text file so that i can make all the plots that i want later
    with open(os.path.join('losses', model_name + '.txt'), 'w') as file:
        file.write(str(train_losses) + '\n')
        file.write(str(test_losses) + '\n')
        file.write(str(perplexity_values))

    # Save the model
    torch.save(model.state_dict(), os.path.join('models', model_name + '.pth'))
    model.eval()

    # evaluate_bleu_and_rouge(model, tokenizer, device, processor.dev_samples, vocab_size)

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
        main(d_model=384, num_heads=8, d_ff=1536, num_layers=6, tokenizer=SubwordTokenizer(), model_name="small_word")