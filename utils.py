import random
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch.nn.functional as F
import re
from tokenizers_classes	 import CharacterLevelTokenizer, SubwordTokenizer


def evaluate_bleu_characters(generated_sequence, ground_truth):
    # Use SmoothingFunction to avoid zero BLEU scores due to lack of n-gram matches
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([list(ground_truth)], list(generated_sequence), smoothing_function=smoothing_function)
    return bleu_score


def evaluate_rouge_characters(generated_sequence, ground_truth):
    rouge = Rouge()
    generated_str = ''.join(generated_sequence)
    ground_truth_str = ''.join(ground_truth)
    rouge_score = rouge.get_scores(generated_str, ground_truth_str)
    return rouge_score[0]['rouge-l']['f']


def evaluate_bleu_subwords(generated_continuation, ground_truth, tokenizer):
    generated_tokens = tokenizer.detokenize(generated_continuation)
    ground_truth_tokens = tokenizer.detokenize(ground_truth)
    generated_tokens = generated_tokens.split()
    ground_truth_tokens = ground_truth_tokens.split()

    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([ground_truth_tokens], generated_tokens, smoothing_function=smoothie)

    return bleu_score


def evaluate_rouge_subwords(generated_continuation, ground_truth, tokenizer):
    generated_tokens = tokenizer.detokenize(generated_continuation)
    ground_truth_tokens = tokenizer.detokenize(ground_truth)

    if not generated_tokens.strip() or generated_tokens.strip() == "<EOS>" or \
        not re.search(r'\w', generated_tokens) or not re.search(r'\w', ground_truth_tokens) or \
        ground_truth_tokens.strip() == "<EOS>" or not ground_truth_tokens.strip():
        return 0.0

    rouge = Rouge()
    scores = rouge.get_scores(generated_tokens, ground_truth_tokens, avg=True)

    return scores['rouge-l']['f']


def evaluate_bleu_and_rouge(model, tokenizer, device, dev_samples, vocab_size, context_size=5, max_length=1024, num_samples=200):
    bleu_scores = []
    rouge_scores = []

    rouge = Rouge()

    # Subsample num_samples samples from the dev set
    random.seed(100)
    sampled_dev_set = random.sample(dev_samples, num_samples)

    for sample in sampled_dev_set:
        words = sample.split()
        context_words = words[:context_size]
        initial_string = " ".join(context_words)
        sample = tokenizer.tokenize(sample)
        context = tokenizer.tokenize(initial_string)
        initial_string = tokenizer.detokenize(context)
        generated_sequence = generate_sequence(model, device, tokenizer, initial_string, vocab_size, max_length)

        generated_continuation = generated_sequence[len(tokenizer.detokenize(context)):]
        generated_continuation = tokenizer.tokenize(generated_continuation)[1:]
        ground_truth = sample[len(context):]

        if type(tokenizer) == CharacterLevelTokenizer:
            bleu_score = evaluate_bleu_characters(generated_continuation, ground_truth)
        else:
            bleu_score = evaluate_bleu_subwords(generated_continuation, ground_truth, tokenizer)
        bleu_scores.append(bleu_score)

        if type(tokenizer) == CharacterLevelTokenizer:
            rouge_score = evaluate_rouge_characters(generated_continuation, ground_truth)
        else:
            rouge_score = evaluate_rouge_subwords(generated_continuation, ground_truth, tokenizer)
        rouge_scores.append(rouge_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)

    print(f'Average BLEU score: {avg_bleu_score:.4f}')
    print(f'Average ROUGE-L score: {avg_rouge_score:.4f}')

    return avg_bleu_score, avg_rouge_score


def generate_sequence(model, device, tokenizer, context, vocab_size, max_length=512, context_size=256, top_k=10):
    context_tokens = tokenizer.tokenize(context)

    if type(tokenizer) == CharacterLevelTokenizer:
        context_ids = torch.tensor([tokenizer.vocab[token] for token in context_tokens]).unsqueeze(0).to(device)
    elif type(tokenizer) == SubwordTokenizer:
        context_ids = torch.tensor(context_tokens).unsqueeze(0).to(device)

    generated_sequence = context_tokens[:-1]

    for _ in range(max_length):
        logits = model(context_ids)
        logits = logits[:, -1, :]

        # Apply top-k sampling
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            top_k_probs = F.softmax(top_k_values, dim=-1)

            top_k_indices = top_k_indices.detach().cpu().numpy()[0]
            top_k_probs = top_k_probs.detach().cpu().numpy()[0]
            next_token_id = random.choices(top_k_indices, top_k_probs)[0]
        else:
            # Otherwise use argmax
            next_token_id = torch.argmax(logits, dim=-1).item()

        if type(tokenizer) == CharacterLevelTokenizer:
            next_token = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(next_token_id)]
        elif type(tokenizer) == SubwordTokenizer:
            next_token = next_token_id

        generated_sequence.append(next_token)
        context_ids = torch.cat((context_ids, torch.tensor([[next_token_id]]).to(device)), dim=1)

        if context_ids.size(1) > context_size:
            context_ids = context_ids[:, 1:]

        # Stop if <EOS> token is generated
        if next_token == tokenizer.end_token:
            break

    generated_sequence = tokenizer.detokenize(generated_sequence)
    return generated_sequence
