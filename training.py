import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from tqdm import tqdm

from transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
BATCH_SIZE = 16  # Note: This is not directly used in the batching logic, which is sequence-length based.
NUM_EPOCHS = 10
EMBED_SIZE = 512
NUM_LAYERS = 6
HEADS = 8
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 100 # Sequence length for processing data

# These will be set dynamically from the tokenizer in main()
SRC_PAD_IDX = None
TRG_PAD_IDX = None
SRC_VOCAB_SIZE = None
TRG_VOCAB_SIZE = None

def get_batch(source_data, i, bptt):
    """
    Generates a batch of data for language modeling.
    The target is the source sequence shifted by one token.
    """
    seq_len = min(bptt, len(source_data) - 1 - i)
    data = source_data[i:i+seq_len]
    target = source_data[i+1:i+1+seq_len].reshape(-1)
    return data, target

def initialize_weights(m):
    """Initializes model weights with Xavier uniform distribution."""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def main():
    """Main function to orchestrate dataset loading, training, and evaluation."""
    global SRC_PAD_IDX, TRG_PAD_IDX, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE

    # 1. Load the WikiText-103 dataset
    print("Loading WikiText-103 dataset...")
    wikitext = load_dataset('wikitext', 'wikitext-103-v1')

    # 2. Train a new tokenizer or load an existing one
    tokenizer_path = 'wikitext_tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print("Training a new BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"])

        # Create an iterator for the trainer
        def text_iterator():
            for item in wikitext['train']['text']:
                if item.strip(): # Ensure non-empty strings
                    yield item

        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}")
    else:
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(tokenizer_path)

    # 3. Set global constants from the loaded tokenizer
    SRC_PAD_IDX = tokenizer.token_to_id("<pad>")
    TRG_PAD_IDX = tokenizer.token_to_id("<pad>")
    SRC_VOCAB_SIZE = tokenizer.get_vocab_size()
    TRG_VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {SRC_VOCAB_SIZE}, Padding Index: {SRC_PAD_IDX}")


    # 4. Define data processing function using the tokenizer
    def data_process(raw_text_iter, seq_len):
        """
        Tokenizes raw text, concatenates into a single tensor,
        and reshapes it into batches of shape [batch_size, seq_len].
        """
        print("Processing data...")
        # Encode all text and filter out empty sequences
        data = [torch.tensor(tokenizer.encode(item).ids, dtype=torch.long) for item in tqdm(raw_text_iter) if item.strip()]
        # Concatenate all tokenized tensors into one long sequence
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        # Determine number of batches and trim excess data
        num_batches = data.size(0) // seq_len
        data = data.narrow(0, 0, num_batches * seq_len)
        # Reshape into [num_batches, seq_len]
        data = data.view(num_batches, seq_len).contiguous()
        return data.to(DEVICE) # Move data to the target device

    train_data = data_process(wikitext['train']['text'], MAX_LENGTH)
    val_data = data_process(wikitext['validation']['text'], MAX_LENGTH)

    # 5. Initialize Model, Optimizer, and Loss Function
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        embed_size=EMBED_SIZE,
        num_layers=NUM_LAYERS,
        forward_expansion=FORWARD_EXPANSION,
        heads=HEADS,
        dropout=DROPOUT,
        device=DEVICE,
        max_length=MAX_LENGTH
    ).to(DEVICE)
    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)

    # --- Training and Evaluation Functions ---
    def train_fn(model, data_source):
        model.train()
        total_loss = 0.
        # Iterate over the data source which is already shaped [num_batches, seq_len]
        for i in tqdm(range(data_source.size(0)), desc="Training"):
            src = data_source[i, :].unsqueeze(0) # Get a single sequence, add batch dim
            trg = data_source[i, :].unsqueeze(0)

            # In language modeling, the input to the decoder is the same as the encoder
            # The model's internal masking handles predicting the next token.
            # The actual target for the loss is handled by shifting inside the loss calculation.
            optimizer.zero_grad()
            output = model(src, trg[:, :-1]) # Target input is shifted right

            # Reshape for loss calculation
            output_reshaped = output.contiguous().view(-1, TRG_VOCAB_SIZE)
            trg_for_loss = trg[:, 1:].contiguous().view(-1) # Ground truth is shifted left

            loss = criterion(output_reshaped, trg_for_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / data_source.size(0)

    def evaluate_fn(model, data_source):
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i in tqdm(range(data_source.size(0)), desc="Evaluating"):
                src = data_source[i, :].unsqueeze(0)
                trg = data_source[i, :].unsqueeze(0)

                output = model(src, trg[:, :-1])
                output_reshaped = output.contiguous().view(-1, TRG_VOCAB_SIZE)
                trg_for_loss = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output_reshaped, trg_for_loss)
                total_loss += loss.item()

        return total_loss / data_source.size(0)

    # 6. Main Training Loop
    best_val_loss = float('inf')
    print("\nStarting training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss = train_fn(model, train_data)
        val_loss = evaluate_fn(model, val_data)

        epoch_time = time.time() - epoch_start_time

        print("-" * 89)
        print(f'| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | '
              f'train loss {train_loss:.3f} | train ppl {math.exp(train_loss):8.2f} | '
              f'val loss {val_loss:.3f} | val ppl {math.exp(val_loss):8.2f}')
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_wikitext103.pt')
            print("Saved best model state to 'transformer_wikitext103.pt'")

if __name__ == "__main__":
    main()
