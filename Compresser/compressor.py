#!/usr/bin/env python3
"""
LLM-based Text Compressor using DistilGPT2 with Arithmetic Coding

Key insight: Use LLM's next-token probabilities as the probability model for arithmetic coding.
Tokens the LLM predicts confidently get encoded in fewer bits, achieving compression.

Performance: KV caching for 10-100x speedup, single-token forward passes after first token.
"""

import time
import torch
import zlib
import pickle
import os
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

try:
    import constriction
except ImportError:
    print("Warning: constriction not available, installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "constriction"])
    import constriction


class LLMTextCompressor:
    def __init__(self, model_name: str = None, context_window: int = 512):
        self.model_name = model_name or "distilgpt2"
        self.context_window = context_window

        print(f"Loading tokenizer and model: {self.model_name}")
        print(f"Context window: {context_window} tokens")

        # CPU only — MPS has dtype casting issues with constriction
        self.device = torch.device("cpu")
        print("Using device: CPU")

        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(cpu_count)
        print(f"Using {cpu_count} CPU threads")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model in fp32...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
        ).to(self.device)

        self.model.eval()
        print(f"Model loaded: {self.model_name}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_token_probabilities(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            return probs, outputs.past_key_values

    def _probs_to_categorical(self, probs: torch.Tensor):
        """
        Convert probability tensor to constriction Categorical model.
        constriction 0.4.x requires a 1D float32 numpy array (NOT 2D).
        """
        p = probs.squeeze().cpu().detach().numpy().astype(np.float32)
        p = np.abs(p)    # remove any negative float errors
        p = p / p.sum()  # renormalize to exactly 1.0
        # must be 1D — 2D breaks constriction 0.4.x
        return constriction.stream.model.Categorical(p, lazy=False)

    def encode_token(self, token: int, probs: torch.Tensor, coder) -> None:
        model = self._probs_to_categorical(probs)
        coder.encode(np.array([token], dtype=np.int32), model)

    def decode_token(self, probs: torch.Tensor, coder) -> int:
        model = self._probs_to_categorical(probs)
        return int(coder.decode(model))

    def compress(self, text: str) -> bytes:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        original_length = len(tokens)
        print(f"Compressing {len(text)} characters -> {original_length} tokens")

        encoder = constriction.stream.queue.RangeEncoder()
        past_key_values = None
        start_time = time.time()

        with tqdm(total=original_length, desc="Compressing") as pbar:
            for i, token in enumerate(tokens):
                if i == 0:
                    # Seed with bos_token to predict first token
                    input_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)
                else:
                    # Pass only the previous token
                    input_ids = torch.tensor([[tokens[i - 1]]], dtype=torch.long)

                probs, past_key_values = self.get_token_probabilities(input_ids, past_key_values)
                self.encode_token(token, probs, encoder)
                pbar.update(1)

        compress_time = time.time() - start_time
        print(f"Compression speed: {original_length / compress_time:.2f} tokens/second")

        compressed_data = encoder.get_compressed()
        return pickle.dumps((compressed_data, original_length))

    def decompress(self, compressed_bytes: bytes) -> str:
        compressed_data, token_count = pickle.loads(compressed_bytes)

        decoder = constriction.stream.queue.RangeDecoder(compressed_data)
        decoded_tokens = []
        past_key_values = None

        print(f"Decompressing -> {token_count} tokens")
        start_time = time.time()

        with tqdm(total=token_count, desc="Decompressing") as pbar:
            for i in range(token_count):
                if i == 0:
                    # Must match compress exactly: seed with first token of the sequence
                    # We don't know the first token yet, so use bos to get probs for token 0
                    input_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)
                else:
                    # Pass only the previously decoded token
                    input_ids = torch.tensor([[decoded_tokens[i - 1]]], dtype=torch.long)

                probs, past_key_values = self.get_token_probabilities(input_ids, past_key_values)
                decoded_token = self.decode_token(probs, decoder)
                decoded_tokens.append(decoded_token)
                pbar.update(1)

        decompress_time = time.time() - start_time
        print(f"Decompression speed: {token_count / decompress_time:.2f} tokens/second")

        return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)


def benchmark(text: str, compressor: LLMTextCompressor, test_name: str):
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {test_name}")
    print(f"{'='*70}")

    original_bytes = len(text.encode('utf-8'))
    token_count = len(compressor.tokenizer.encode(text, add_special_tokens=False))
    print(f"Input: {original_bytes} bytes ({token_count} tokens)")

    # LLM compress
    start = time.time()
    compressed = compressor.compress(text)
    compress_time = time.time() - start
    compressed_size = len(compressed)
    llm_ratio = (compressed_size / original_bytes) * 100
    llm_tps = token_count / compress_time

    # LLM decompress
    start = time.time()
    decompressed = compressor.decompress(compressed)
    decompress_time = time.time() - start
    decompress_tps = token_count / decompress_time

    # Lossless check
    if decompressed == text:
        print("\n✓ PASS: Lossless reconstruction verified!")
    else:
        print("\n✗ FAIL: Reconstruction mismatch")
        print(f"  Original:     {repr(text[:80])}")
        print(f"  Decompressed: {repr(decompressed[:80])}")
        return None

    # Zlib
    start = time.time()
    zlib_compressed = zlib.compress(text.encode('utf-8'), level=9)
    zlib_time = time.time() - start
    zlib_size = len(zlib_compressed)
    zlib_ratio = (zlib_size / original_bytes) * 100

    # Table
    print(f"\n{'Method':<10} {'Size':<10} {'Ratio':<10} {'Time':<12} {'Speed'}")
    print(f"{'-'*58}")
    print(f"{'LLM':<10} {compressed_size:<10} {llm_ratio:<9.1f}% {compress_time:<11.2f}s {llm_tps:.2f} tok/s")
    print(f"{'Zlib':<10} {zlib_size:<10} {zlib_ratio:<9.1f}% {zlib_time:<11.4f}s N/A")

    return {
        'llm_ratio': llm_ratio, 'zlib_ratio': zlib_ratio,
        'llm_size': compressed_size, 'zlib_size': zlib_size,
        'llm_tps': llm_tps, 'decompress_tps': decompress_tps,
        'token_count': token_count, 'original_bytes': original_bytes,
    }


def main():
    compressor = LLMTextCompressor(model_name="distilgpt2", context_window=512)

    test_cases = [
        ("English Prose", """
The quick brown fox jumps over the lazy dog. This is a classic pangram that contains
every letter of the English alphabet at least once. It is often used for testing
fonts and keyboards because it displays all characters in a short, memorable phrase.
The sentence has been used since the late 19th century and continues to be popular
today for various applications including typing practice and font previews.
In addition to its practical uses, pangrams like this one are simply fun to read
and demonstrate the flexibility and richness of the English language.
        """.strip()),

        ("Python Code", """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])
    return seq

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"added {a} + {b} = {result}")
        return result
        """.strip()),

        ("JSON Data", """
{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "active": true},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "active": false},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": true},
    {"id": 4, "name": "Diana", "email": "diana@example.com", "active": true}
  ],
  "products": [
    {"id": 101, "name": "Widget", "price": 19.99, "in_stock": true},
    {"id": 102, "name": "Gadget", "price": 29.99, "in_stock": false},
    {"id": 103, "name": "Doohickey", "price": 9.99, "in_stock": true}
  ]
}
        """.strip()),
    ]

    results = []
    all_passed = True

    for name, text in test_cases:
        result = benchmark(text, compressor, name)
        results.append((name, result))
        if result is None:
            all_passed = False

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("✅ ALL TESTS PASSED\n" if all_passed else "❌ SOME TESTS FAILED\n")

    for i, (name, result) in enumerate(results, 1):
        if result is None:
            continue
        print(f"{i}. {name}")
        print(f"   Tokens:  {result['token_count']}")
        print(f"   LLM:     {result['llm_size']} bytes ({result['llm_ratio']:.1f}%) @ {result['llm_tps']:.2f} tok/s")
        print(f"   Zlib:    {result['zlib_size']} bytes ({result['zlib_ratio']:.1f}%)")
        print()


if __name__ == "__main__":
    main()
