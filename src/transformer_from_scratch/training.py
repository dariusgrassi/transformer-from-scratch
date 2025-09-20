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
from .transformer import Transformer

# --- Hyperparameters and Constants ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
EMBED_SIZE = 128
NUM_LAYERS = 2
HEADS = 4
FORWARD_EXPANSION = 2
DROPOUT = 0.1
MAX_LENGTH = 512

# --- File Paths for Local Storage ---
ARTIFACTS_PATH = 'artifacts'
TOKENIZER_PATH = os.path.join(ARTIFACTS_PATH, 'wikitext_tokenizer.json')
TRAIN_DATA_PATH = os.path.join(ARTIFACTS_PATH, 'train_data.pt')
VAL_DATA_PATH = os.path.join(ARTIFACTS_PATH, 'val_data.pt')
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_PATH, 'transformer_wikitext103.pt')

# --- Global variables ---
SRC_PAD_IDX, TRG_PAD_IDX, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE = None, None, None, None

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def main():
    global SRC_PAD_IDX, TRG_PAD_IDX, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE

    # 1. Create local artifacts directory
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    print(f"Artifacts will be saved to/loaded from: {ARTIFACTS_PATH}")

    # 2. Load or Train Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    else:
        print("Training a new BPE tokenizer...")
        wikitext_for_tokenizer = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"])
        def text_iterator():
            for item in wikitext_for_tokenizer:
                if item['text'].strip(): yield item['text']
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        tokenizer.save(TOKENIZER_PATH)
        print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")

    SRC_PAD_IDX = tokenizer.token_to_id("<pad>")
    TRG_PAD_IDX = tokenizer.token_to_id("<pad>")
    SRC_VOCAB_SIZE = tokenizer.get_vocab_size()
    TRG_VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {SRC_VOCAB_SIZE}, Padding Index: {SRC_PAD_IDX}")

    # 3. Load or Process Data
    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(VAL_DATA_PATH):
        print(f"Loading processed data from {ARTIFACTS_PATH}...")
        train_data = torch.load(TRAIN_DATA_PATH).to(DEVICE)
        val_data = torch.load(VAL_DATA_PATH).to(DEVICE)
    else:
        print("Processing data from scratch...")
        wikitext = load_dataset('wikitext', 'wikitext-103-v1')
        def data_process(raw_text_iter, seq_len):
            data = [torch.tensor(tokenizer.encode(item['text']).ids, dtype=torch.long) for item in tqdm(raw_text_iter) if item['text'].strip()]
            data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
            num_sequences = data.size(0) // seq_len
            data = data.narrow(0, 0, num_sequences * seq_len)
            data = data.view(num_sequences, seq_len).contiguous()
            return data

        train_data = data_process(wikitext['train'], MAX_LENGTH)
        val_data = data_process(wikitext['validation'], MAX_LENGTH)

        print(f"Saving processed data to {ARTIFACTS_PATH}...")
        torch.save(train_data, TRAIN_DATA_PATH)
        torch.save(val_data, VAL_DATA_PATH)
        train_data = train_data.to(DEVICE)
        val_data = val_data.to(DEVICE)

    # 4. Initialize Model, Optimizer, and Loss Function
    print(f"Initializing model on {DEVICE}...")
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

    def train_fn(model, data_source):
        model.train()
        total_loss = 0.
        num_batches = 0
        for i in tqdm(range(0, data_source.size(0), BATCH_SIZE), desc="Training"):
            batch = data_source[i:i + BATCH_SIZE]
            src, trg = batch, batch
            optimizer.zero_grad(set_to_none=True)
            output = model(src, trg[:, :-1])
            output_reshaped = output.contiguous().view(-1, TRG_VOCAB_SIZE)
            trg_for_loss = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshaped, trg_for_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def evaluate_fn(model, data_source):
        model.eval()
        total_loss = 0.
        num_batches = 0
        with torch.no_grad():
            for i in tqdm(range(0, data_source.size(0), BATCH_SIZE), desc="Evaluating"):
                batch = data_source[i:i + BATCH_SIZE]
                src, trg = batch, batch
                output = model(src, trg[:, :-1])
                output_reshaped = output.contiguous().view(-1, TRG_VOCAB_SIZE)
                trg_for_loss = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output_reshaped, trg_for_loss)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    # 5. Main Training Loop
    best_val_loss = float('inf')
    print(f"\nStarting training with batch size {BATCH_SIZE}...")

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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model state to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

