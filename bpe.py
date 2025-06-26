import regex as re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        '''i took the latest regex pattern from the tiktoken repo from OpenAI
        and not the one from the og paper'''
        self.pattern = r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = {i: i for i in range(256)}  
        self.merges = {}

    def get_bytes(self, text):
        text = text.encode('utf-8')
        return list(map(int, text))


    def get_pair_freq(text):
        pair_freq = {}
        for i in range(len(text) - 1):
            # pair = text[i : i + 1] 
            '''lists are unhashable and cannot be used as dictionary keys. 
            So we should use a tuple or a single int as the key.'''

            pair = (text[i], text[i + 1])  
            # print(pair)
            if pair in pair_freq:
                pair_freq[pair] += 1
            else:
                pair_freq[pair] = 1
        return pair_freq

    def merge_pair(self, token_ids, pair, new_token_id):
        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                new_token_ids.append(new_token_id)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        return new_token_ids
    
    def train(self, text):
        chunks = re.findall(self.pattern, text)
        
        token_sequences = []
        for chunk in chunks:
            token_sequences.append(self.get_bytes(chunk))
        all_tokens = []
        for token_sequence in token_sequences:
            all_tokens.extend(token_sequence)
        
        print(f"Initial token count: {len(all_tokens)}")
        
        current_vocab_size = 256
        while current_vocab_size < self.vocab_size:
            pair_freq = defaultdict(int)

            for seq in token_sequences:
                seq_pairs = self.get_pair_freq(seq)
                for pair, freq in seq_pairs.items():
                    pair_freq[pair] += freq
            if not pair_freq:
                break

            most_freq = max(pair_freq, key = pair_freq.get)

            if pair_freq[most_freq] < 2:
                break

            new_token_id = current_vocab_size

            token1_bytes = self.vocab[most_freq[0]]
            token2_bytes = self.vocab[most_freq[1]]

            self.vocab[new_token_id] = token1_bytes + token2_bytes

            for i in range(len(token_sequences)):
                token_sequences[i] = self.merge_pair(token_sequences[i], most_freq, new_token_id)
            current_vocab_size += 1
            
            if current_vocab_size % 1000 == 0:
                print(f"Vocab size: {current_vocab_size}, Best pair: {most_freq} -> {new_token_id}")
        
        print(f"Training complete. Final vocab size: {current_vocab_size}")
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}



    def encode(self, text):
        chunks = re.findall(self.pattern, text)
        all_tokens = []
        for chunk in chunks:
            token_ids = self.get_bytes(chunk)
            
            while True:
                pair_freq = self.get_pair_freq(token_ids)

                best_pair = None
                best_merge_id = float('inf')

                for pair in pair_freq:
                    if pair in self.merges and self.merges[pair] < best_merge_id:
                        best_pair = pair
                        best_merge_id = self.merges[pair]
                if best_pair is None:
                    break

                token_ids = self.merge_pair(token_ids, best_pair, self.vocab_size)

            all_tokens.extend(token_ids)
        return all_tokens

    def decode(self, token_ids):
        bytes_seq = b''
        for id in token_ids:
            if id in self.vocab:
                bytes_seq += self.vocab[id]
            else:
                bytes_seq += b"<|unk|>"
        return bytes_seq.decode('utf-8', errors='replace')
    
    def save_vocab(self, filepath):
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for token_id, token_bytes in self.vocab.items():
                try:
                    token_str = token_bytes.decode('utf-8')
                    f.write(f"{token_id}\t{repr(token_str)}\n")
                except UnicodeDecodeError:
                    f.write(f"{token_id}\t{token_bytes}\n")

    












