from decepticon import causal_language_model_loss, DecoderOnlyTransformer
from dataset_processor import ShakespeareDatasetProcessor
from tokenizers_classes import CharacterLevelTokenizer, SubwordTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def generate_sequence(model, tokenizer, start_sequence, context_size=256, max_tokens=1024, sampling_strategy="argmax", top_k=5):
    """
    Generate a sequence using a sliding window context.
    Args:
        model: The trained transformer model.
        tokenizer: Tokenizer for encoding/decoding tokens.
        start_sequence: The initial sequence to seed generation.
        context_size: Maximum number of tokens to attend to.
        max_tokens: Maximum tokens to generate.
        sampling_strategy: "argmax" or "top-k".
        top_k: Number of tokens to consider in top-k sampling.
    Returns:
        Generated sequence as a string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Encode the starting sequence
    context = [tokenizer.token_to_id(token) for token in tokenizer.tokenize(start_sequence)]
    print(context)
    generated_tokens = context.copy()  # Store all generated tokens

    for _ in range(max_tokens):
        # Ensure context fits the context window size
        input_context = torch.tensor([context[-context_size:]], device=device)  # Trim context to max size

        # Generate logits from the model
        with torch.no_grad():
            logits = model(input_context)[:, -1, :]  # Take the last token's logits

        # Apply the sampling strategy
        if sampling_strategy == "argmax":
            next_token_id = torch.argmax(logits, dim=-1).item()
        elif sampling_strategy == "top-k":
            # Select top-k tokens
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            top_k_probs /= top_k_probs.sum()  # Renormalize probabilities
            next_token_id = torch.multinomial(top_k_probs, num_samples=1).item()
            next_token_id = top_k_indices[next_token_id].item()
        else:
            raise ValueError("Invalid sampling strategy. Choose 'argmax' or 'top-k'.")

        # Append the next token to the generated sequence
        context.append(next_token_id)
        generated_tokens.append(next_token_id)

        # Stop if <EOS> is generated
        if next_token_id == tokenizer.token_to_id("<EOS>"):
            break

    # Decode the generated tokens into text
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)
    return generated_text


class TextDataset(Dataset):
    def __init__(self, tokenized_samples, vocab, pad_token_idx):
        self.tokenized_samples = tokenized_samples
        self.vocab = vocab
        self.pad_token_idx = pad_token_idx
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

    all_tokens = [char for tokens in (train_tokens + dev_tokens + eval_tokens) for char in tokens]
    vocab = {char: idx for idx, char in enumerate(set(all_tokens))}
    vocab['<PAD>'] = len(vocab)
    pad_token_idx = vocab['<PAD>']

    # Create the data loaders
    train_dataset = TextDataset(train_tokens, vocab, pad_token_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    dev_dataset = TextDataset(dev_tokens, vocab, pad_token_idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size=50, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))
    eval_dataset = TextDataset(eval_tokens, vocab, pad_token_idx)
    eval_dataloader = DataLoader(eval_dataset, batch_size=50, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, train_dataset.pad_token_idx))

    vocab_size = len(train_dataset.vocab)
    model = DecoderOnlyTransformer(vocab_size=vocab_size, pad_token_idx=train_dataset.pad_token_idx, d_model=384, num_heads=6, d_ff=1536, num_layers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 3
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

    generate_sequence(model, tokenizer, "King Lear: ")

if __name__ == '__main__':
    main()