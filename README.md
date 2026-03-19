# Neural Arithmetic Compression

A neural text compressor that combines Large Language Models (LLMs) with arithmetic coding to achieve lossless text compression. This innovative approach uses the LLM's next-token probabilities as the probability model for arithmetic coding, enabling tokens that the model predicts confidently to be encoded in fewer bits.

Unlike traditional compression algorithms (like zlib/zstd) that rely on statistical patterns in the text, this compressor leverages the predictive power of modern LLMs. The LLM predicts the next token and provides probability distributions over the entire vocabulary. These probabilities are fed into an arithmetic coder, where predicted tokens are encoded using fewer bits. The same LLM is used to generate identical probability distributions (since arithmetic coding doesn't introduce errors), allowing perfect reconstruction of the original text. From testing it is shown that all texts can be recreated with perfect accuarcy. However, for text with irregular patterns (eg: Random strings/dauhwdyfubkjsnfaikbwd), this may not be the case.

//Out-of-distribution text (unpredictable by the model) gets more bits, in-distribution text gets fewer bits - aligning compression efficiency with the model's strengths.//

## Performance and Benchmark Results

The compressor was tested on three different text types, achieving significant compression ratios:

| Text Type | Original Size | LLM Compressed | LLM Ratio | Zlib Ratio | 
|-----------|---------------|----------------|-----------|------------|
| English Prose | 560 bytes | 232 bytes      | **41.4%** | 61.4%      | 
| Python Code | 485 bytes   | 219 bytes      | **45.2%** | 60.6%      |
| JSON Data   | 555 bytes   | 251 bytes      | **45.1%** | 64.5%      | 

*Results based on distilgpt2 model with 512-token context window*

### Speed from testing
- **Compression**: ~45 tokens/second (CPU)
- **Decompression**: ~80 tokens/second (CPU)
- Uses KV caching for speedups after the first token
- Single-token forward passes after the initial sequence

## Installation

### Prerequisites
- Python 3.7+
- CUDA (optional, CPU works well)


### How Compression Works

1. **Tokenization**: Input text is tokenized using the model's tokenizer
2. **Initialization**: Arithmetic coder is initialized
3. **Streaming compression**:
   - For each token, the model predicts next-token probabilities
   - If it's the first token, seed with `bos_token`
   - For subsequent tokens, use only the previous token as input
   - Probability distribution is converted to a categorical model
   - Token is encoded using arithmetic coding
   - Past key-values are cached for efficient prediction
4. **Serialization**: Compressed data and metadata are pickled

### How Decompression Works

1. **Deserialization**: Load compressed data and metadata
2. **Streaming decompression**: The exact reverse of compression
   - Use model to predict token probabilities (same as during compression)
   - Arithmetic decoder recovers the exact token
3. **Decoding**: Tokens are converted back to text

## Compression Results

All compression runs are automatically saved to `compression_runs/` with detailed metadata:

```json
{
  "label": "English Prose",
  "model": "distilgpt2",
  "original_size_bytes": 560,
  "llm_zstd_size_bytes": 232,
  "llm_ratio_percent": 41.43,
  "zlib_ratio_percent": 61.43,
  "compression_tps": 45.56,
  "decompression_tps": 80.67,
  "lossless_verified": true
}
```

Each run includes:
- `compressed.bin`: The compressed data
- `original.txt`: Original text (for reference)
- `decompressed.txt`: Decompressed text (verification)
- `run_info.json`: Complete metadata and statistics

## Model Selection

The default model is **DistilGPT2** (82M parameters), chosen because of:
- Fast inference: Good for CPU compression
- Strong language modeling: Captures text patterns effectively
- Small size: Reasonable model overhead
- Compatible tokenization: Works with standard BPE tokenizer

You can experiment with other models, but not tested. Larger models COULD* Yield Worse Results:

```python
compressor = LLMTextCompressor(model_name="gpt2", context_window=512)
```

## Limitations

- **Slow on CPU**: ~45 tokens/sec (but still practical for many use cases)
- **Fixed vocab size**: Uses model's vocabulary (50K+ tokens for GPT2)
- **Context window**: Limited by model's maximum context length
- **GPU compatibility**: Currently CPU-only (MPS has dtype issues with constriction)

## Here's a Theory: Arithmetic Coding

Arithmetic coding is a form of entropy coding that represents a message as a single number in [0, 1). Unlike Huffman coding (which uses integer-length codes), arithmetic coding assigns fractional bits to symbols, achieving better compression when probabilities are not powers of 2.

The compressor uses the [constriction](https://github.com/botika/constriction) library for robust implementation of range coding (the integer variant of arithmetic coding).

1. Start with interval [0, 1)
2. Divide interval based on symbol probabilities
3. Narrow interval to the sub-interval of the symbol to encode
4. Repeat for each symbol
5. Output any number within the final interval

For the LLM compressor, symbol probabilities come from the neural network's predictions.

## Comparison with Traditional Compression

### Zlib (Deflate)
- **Algorithm**: LZ77 + Huffman coding
- **Advantages**: Fast, widely supported, good for repetitive patterns
- **Limitations**: No semantic understanding, treats all patterns equally

### Neural Compression (This Project)
- **Algorithm**: LLM predictions + Arithmetic coding
- **Advantages**: Semantic understanding, better for natural language
- **Limitations**: Slower, requires ML model

### Why Neural COULD be better

1. **Contextual understanding**: LLM understands language structure and semantics
2. **Better probability estimates**: Neural models learned from massive datasets
3. **Adaptive modeling**: Predictions adapt to content type (code vs. prose vs. JSON)
4. **Low-entropy text**: Predictable content compresses extremely well

## License

MIT License - feel free to use for research or commercial purposes.

## Citation

If you use this compressor in your research, please cite:

```bibtex
@misc{neural_arithmetic_compression,
  title={Neural Arithmetic Compression: LLM-based Text Compression using Arithmetic Coding},
  author={Aspen Wang},
  year={2026},
  url={https://github.com/Solar-cmd/neural-arithmetic-compression}
}
```

## References

- **Arithmetic Coding**: Witten, I. H., Neal, R. M., & Cleary, J. G. (1987). Arithmetic coding for data compression. *Communications of the ACM, 30*(6), 520-540.
- **DistilGPT2**: Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
- **Hugging Face Transformers**: Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP*.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Made with ❤️ and a deep understanding of both compression theory and modern NLP, and 14hrs of work :/
