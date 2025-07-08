# Building a small version of BPE from scratch

A lightweight, from-scratch implementation of Byte Pair Encoding (BPE) tokenization in Python. Built this for learning and understanding how modern tokenizers work under the hood.

## Quick Start

```bash
git clone https://github.com/yourusername/smol-bpe-tokenizer
cd smol-bpe-tokenizer
python train_tokenizer.py
```

Some suggested vocab sizes:

1. Small (2000) - Fast training
2. Medium (5000) - Good balance
3. Large (8000) - Better compression

I used a 10k vocab size.

## Training details

Trained on a 4MB subset of the TinyStories dataset:

- Characters processed: 4,155,082
- Training time: ~75 minutes (10K vocab on M2 Air)
- Memory usage: ~153MB peak
- Compression ratio: 1.00x (character-level for this dataset)

## Outputs

The train.py outputs 2 files:

- outputs/tinystories_vocab.txt - the learned vocabulary
- outputs/training_stats.json - training metrics and performance data


