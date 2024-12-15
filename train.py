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
        self.vocab = {char: idx for idx, char in enumerate(set(c for player, line in tokenized_samples for c in player + line))}
        self.vocab['<PAD>'] = len(self.vocab)
        self.pad_token_idx = self.vocab['<PAD>']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        player_tokens, line_tokens = self.tokenized_samples[idx]
        player_indices = torch.tensor([self.vocab[token] for token in player_tokens])
        line_indices = torch.tensor([self.vocab[token] for token in line_tokens])
        return player_indices, line_indices

def collate_fn(batch, pad_token_idx):
    # Pad sequences to the same length using pad_token_idx
    player_indices, line_indices = zip(*batch)
    player_indices_padded = pad_sequence(player_indices, batch_first=True, padding_value=pad_token_idx)
    line_indices_padded = pad_sequence(line_indices, batch_first=True, padding_value=pad_token_idx)
    return player_indices_padded, line_indices_padded


def main():
    processor = ShakespeareDatasetProcessor('Shakespeare_data.csv')
    tokenizer = CharacterLevelTokenizer()
    processor.load_data()
    samples = processor.process_data()

    processor.split_data(samples)
    train_tokens = [(tokenizer.tokenize(player), tokenizer.tokenize(line)) for player, line in processor.train_samples]
    dev_tokens = [(tokenizer.tokenize(player), tokenizer.tokenize(line)) for player, line in processor.dev_samples]
    eval_tokens = [(tokenizer.tokenize(player), tokenizer.tokenize(line)) for player, line in processor.eval_samples]

    # Create the data loaders
    train_dataset = TextDataset(train_tokens, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=15, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    dev_dataset = TextDataset(dev_tokens, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=15, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    eval_dataset = TextDataset(eval_tokens, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=15, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))

    vocab_size = len(train_dataset.vocab)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=384, num_heads=6, d_ff=1536, num_layers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        for batch_idx, (player_indices, line_indices) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            player_indices, line_indices = player_indices.to(device), line_indices.to(device)
            inputs = torch.cat((player_indices, line_indices), dim=1)
            logits = model(inputs)

            loss = causal_language_model_loss(logits, inputs, train_dataset.pad_token_idx)
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

        with torch.no_grad():
            for batch_idx, (player_indices, line_indices) in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Dev PPL"):
                player_indices, line_indices = player_indices.to(device), line_indices.to(device)
                inputs = torch.cat((player_indices, line_indices), dim=1)
                logits = model(inputs)

                for i in range(line_indices.size(1) - 1):
                    target_tokens = line_indices[:, i+1]
                    predicted_logits = logits[:, i, :]
                    loss = nn.CrossEntropyLoss()(predicted_logits, target_tokens)
                    total_log_likelihood += loss.item() * target_tokens.size(0)
                    total_token_count += target_tokens.size(0)

        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_token_count
        perplexity = torch.exp(torch.tensor(avg_log_likelihood))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Dev Perplexity: {perplexity.item():.3f}")


    # Evaluate the model
    model.eval()
    total_log_likelihood = 0.0
    total_token_count = 0

    with torch.no_grad():
        for batch_idx, (player_indices, line_indices) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Test PPL"):
            player_indices, line_indices = player_indices.to(device), line_indices.to(device)

            inputs = torch.cat((player_indices, line_indices), dim=1)
            logits = model(inputs)

            # Calculate the next token prediction loss
            for i in range(line_indices.size(1) - 1):  # For each token in the sequence (except last)
                target_tokens = line_indices[:, i+1]  # Target is the next token
                predicted_logits = logits[:, i, :]    # Logits for the current position
                loss = nn.CrossEntropyLoss()(predicted_logits, target_tokens)
                total_log_likelihood += loss.item() * target_tokens.size(0)  # Multiply by batch size
                total_token_count += target_tokens.size(0)

    # Calculate final test perplexity
    avg_log_likelihood = total_log_likelihood / total_token_count
    test_perplexity = torch.exp(torch.tensor(avg_log_likelihood))


if __name__ == '__main__':
    main()