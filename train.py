import os
import time
import json
import requests
import gzip
from pathlib import Path
import psutil
from bpe import BPETokenizer
from datasets import load_dataset
import requests
from datasets import load_dataset

def download_tinystories_train():

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    subset = dataset.select(range(5000))
    # subset = dataset 

    os.makedirs("data", exist_ok=True)
    output_path = "data/TinyStories-train.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for example in subset:
            f.write(example["text"].strip() + "\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved the subset to {output_path} ({size_mb:.1f}MB)")
    return output_path

def preprocess_tinystories(input_path, max_size_mb=None):
  
    output_path = input_path.replace('.txt', '_clean.txt')
    
    if os.path.exists(output_path):
        print(f"Preprocessed file already exists: {output_path}")
        return output_path
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:
            clean_lines.append(line)
    
    clean_content = '\n'.join(clean_lines)
    
    if max_size_mb:
        max_chars = max_size_mb * 1024 * 1024  
        if len(clean_content) > max_chars:
            clean_content = clean_content[:max_chars]
            last_period = clean_content.rfind('.')
            if last_period > max_chars - 1000:  
                clean_content = clean_content[:last_period + 1]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"preprocessed file: {size_mb:.1f}MB")
    return output_path

def monitor_system() :
  
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent()
    return memory_mb, cpu_percent

def train_tinystories_bpe(dataset_path, vocab_size=8000):

    print(f"Vocabulary Size: {vocab_size:,}")
    print(f"Dataset: {dataset_path}")
    
    print("Loading data...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset stats:")
    print(f"  Characters: {len(text):,}")
    print(f"  Size: {len(text)/(1024*1024):.1f}MB")
    print(f"  Lines: {text.count(chr(10)):,}")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    start_time = time.time()
    start_memory, _ = monitor_system()    
    print(f"training started...")
    try:
        tokenizer.train(text)
        
        end_time = time.time()
        end_memory, _ = monitor_system()
        
        training_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"\nTraining Complete!")
        print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
        print(f"Memory used: {memory_used:.1f}MB")
        print(f"Final vocab size: {len(tokenizer.vocab):,}")
        print(f"Learned merges: {len(tokenizer.merges):,}")
        
        return tokenizer, {
            'training_time': training_time,
            'memory_used': memory_used,
            'final_vocab_size': len(tokenizer.vocab)
        }
        
    except Exception as e:
        print(f"Training failed due to error: {e}")
        return None, None

def evaluate_tokenizer(tokenizer, test_stories=None):

    if test_stories is None:
        test_stories = [
            "Once upon a time, there was a little girl who loved to play with her toys.",
            "The cat sat on the mat and looked out the window at the birds.",
            "Tom and his friends went to the park to play with a big red ball.",
            "The happy dog wagged its tail when it saw its favorite treat.",
            "In the magical forest, all the animals lived together peacefully.",
        ]
    
    total_chars = 0
    total_tokens = 0
    
    print("eval... Tokenization Examples:")
    
    for i, story in enumerate(test_stories, 1):
        print(f"\n Story {i}:")
        print(f"Original: '{story}'")
        
        tokens = tokenizer.encode(story)
        decoded = tokenizer.decode(tokens)
        total_chars += len(story)
        total_tokens += len(tokens)
        compression = len(story) / len(tokens) if len(tokens) > 0 else 0
        
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Decoded: '{decoded}'")
        print(f"Match: {'perfect' if story == decoded else 'mismatches found -_- lol'}")
        print(f"Compression: {compression:.2f}x")
    
    overall_compression = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"\nOverall Performance:")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average compression: {overall_compression:.2f}x")
    
    return overall_compression

def show_learned_vocabulary(tokenizer, num_examples=20):
  
    print("Learned Vocabulary Examples")

    base_tokens = [(i, tokenizer.vocab[i]) for i in range(256)]
    learned_tokens = []
    
    for token_id, token_bytes in tokenizer.vocab.items():
        if token_id >= 256:
            try:
                token_str = token_bytes.decode('utf-8')
                learned_tokens.append((token_id, token_str, len(token_str)))
            except UnicodeDecodeError:
                learned_tokens.append((token_id, f"<bytes: {token_bytes}>", len(token_bytes)))
    
    learned_tokens.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Total vocabulary: {len(tokenizer.vocab):,}")
    print(f"Base tokens (bytes): 256")
    print(f"Learned tokens: {len(learned_tokens):}")
    
    print(f"\nTop {num_examples} learned tokens (by length):")
    for i, (token_id, token_str, length) in enumerate(learned_tokens[:num_examples]):
        print(f"  {token_id:5d}: '{token_str}' ({length} chars)")
    
    print(f"\nCommon story words found:")
    story_words = ['the', 'and', 'was', 'had', 'her', 'his', 'she', 'he', 'they', 'once', 'time', 'little']
    found_words = []
    
    for token_id, token_str, length in learned_tokens:
        for word in story_words:
            if word in token_str.lower() and len(token_str.strip()) >= len(word):
                found_words.append(f"'{token_str}' (contains '{word}')")
                break
    
    for word in found_words[:10]:
        print(word)

def save_tokenizer_artifacts(tokenizer, stats, output_dir="outputs"):

    print(f"\nSaving tokenizer artifacts to {output_dir}/")    
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "tinystories_vocab.txt")
    tokenizer.save_vocab(vocab_path)
    
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # merges_path = os.path.join(output_dir, "merge_rules.txt")
    # with open(merges_path, 'w', encoding='utf-8') as f:
    #     f.write("Format: token1_id,token2_id -> new_token_id\n\n")
        
    #     for (t1, t2), new_id in tokenizer.merges.items():
    #         try:
    #             t1_str = tokenizer.vocab[t1].decode('utf-8')
    #             t2_str = tokenizer.vocab[t2].decode('utf-8')
    #             new_str = tokenizer.vocab[new_id].decode('utf-8')
    #             f.write(f"{t1},{t2} -> {new_id}  # '{t1_str}' + '{t2_str}' = '{new_str}'\n")
    #         except UnicodeDecodeError:
    #             f.write(f"{t1},{t2} -> {new_id}  # <binary tokens>\n")

    print(f"vocab: {vocab_path}")
    print(f"stats: {stats_path}")
    
    return vocab_path

def main():

    try:
        dataset_path = download_tinystories_train()
    except Exception as e:
        print(f"Using sample dataset due to error: {e}")
    
    print(f"\nCurrent dataset size: {os.path.getsize(dataset_path)/(1024*1024):.1f}MB")
    
    if os.path.getsize(dataset_path) > 100 * 1024 * 1024:  
        clean_path = preprocess_tinystories(dataset_path, max_size_mb=50)
    else:
        clean_path = preprocess_tinystories(dataset_path)
    
    vocab_size = 10000
    tokenizer, stats = train_tinystories_bpe(clean_path, vocab_size)
    
    if tokenizer is None:
        print("error in tokenizer")
        return
    
    compression = evaluate_tokenizer(tokenizer)
    stats['compression_ratio'] = compression

    show_learned_vocabulary(tokenizer)
    
    save_tokenizer_artifacts(tokenizer, stats)
    
    print("training complete!")
    print(f"Tokenizer trained and is ready with {len(tokenizer.vocab):,} tokens")
    print(f"Training time {stats['training_time']/60:.1f} minutes")
    print(f"Average compression: {compression:.2f}x")

if __name__ == "__main__":
    main()
