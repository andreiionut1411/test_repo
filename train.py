from decepticon import causal_language_model_loss, DecoderOnlyTransformer
from dataset_processor import ShakespeareDatasetProcessor
from tokenizers_classes import CharacterLevelTokenizer, SubwordTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    def __init__(self, tokenized_samples, tokenizer):
        self.tokenized_samples = tokenized_samples
        self.tokenizer = tokenizer
        self.vocab = {char: idx for idx, char in enumerate(set(c for tokens in tokenized_samples for c in tokens))}
        self.vocab['<PAD>'] = len(self.vocab)
        self.pad_token_idx = self.vocab['<PAD>']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        token_indices = torch.tensor([self.vocab[token] for token in self.tokenized_samples[idx]])
        return token_indices

def collate_fn(batch, pad_token_idx):
    inputs_padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_idx)
    targets_padded = inputs_padded.clone()
    return inputs_padded, targets_padded


def main():
    processor = ShakespeareDatasetProcessor('Shakespeare_data.csv')
    tokenizer = CharacterLevelTokenizer()
    processor.load_data()
    samples = processor.process_data()
    samples = [' '.join(sample) for sample in samples]

    processor.split_data(samples)
    train_tokens = [tokenizer.tokenize(sample) for sample in processor.train_samples]
    dev_tokens = [tokenizer.tokenize(sample) for sample in processor.dev_samples]
    eval_tokens = [tokenizer.tokenize(sample) for sample in processor.eval_samples]

    # Create the data loaders
    train_dataset = TextDataset(train_tokens, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    dev_dataset = TextDataset(dev_tokens, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=50, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    eval_dataset = TextDataset(eval_tokens, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=50, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))

    vocab_size = len(train_dataset.vocab)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, pad_token_idx=train_dataset.pad_token_idx, d_model=384, num_heads=6, d_ff=1536, num_layers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 10
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

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
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


if __name__ == '__main__':
    main()