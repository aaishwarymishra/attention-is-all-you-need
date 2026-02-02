# Attention Is All You Need

A complete implementation of the Transformer architecture from scratch using TensorFlow, based on the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Overview

This repository contains a from-scratch implementation of the Transformer model, demonstrating the key components that revolutionized natural language processing:

- **Positional Encoding** - Adding positional information to input embeddings
- **Multi-Head Attention** - The core mechanism for capturing relationships in sequences
- **Encoder-Decoder Architecture** - Complete transformer blocks for sequence-to-sequence tasks
- **Feed-Forward Networks** - Position-wise fully connected layers

## Features

- Custom Positional Encoding implementation
- Multi-Head Attention mechanism
- Cross-Attention for encoder-decoder interaction
- Causal (Masked) Self-Attention for decoder
- Complete Encoder and Decoder blocks
- Full Transformer model for sequence-to-sequence tasks
- Visualization of positional encodings

## Requirements

```bash
tensorflow>=2.x
numpy
matplotlib
```

## Architecture Components

### Positional Encoding

Implements sinusoidal positional encoding to inject sequence position information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Multi-Head Attention

The core attention mechanism that allows the model to focus on different positions:

- Splits embeddings into multiple heads
- Computes scaled dot-product attention
- Concatenates and projects heads back

### Transformer Blocks

**Encoder Block:**
- Multi-head self-attention
- Feed-forward network
- Layer normalization and residual connections

**Decoder Block:**
- Masked multi-head self-attention
- Cross-attention with encoder output
- Feed-forward network
- Layer normalization and residual connections

## 📓 Usage

Open and run the `Transformer.ipynb` notebook in Google Colab or Jupyter:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaishwarymishra/attention-is-all-you-need/blob/main/Transformer.ipynb)

### Quick Example

```python
# Create positional encoding
pos_encoding = positional_encoding(length=100, depth=512)

# Initialize positional embedding layer
pos_embed = PositionalEmbedding(vocab_size=10000, d_model=512)

# Multi-head attention
attention = MultiHeadAttention(num_heads=8, key_dim=512)

# Complete transformer
transformer = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=10000,
    target_vocab_size=10000
)
```

## Model Training

The notebook includes training on a translation task with:
- Custom learning rate schedule (warmup + decay)
- Adam optimizer
- Sparse categorical cross-entropy loss
- Teacher forcing during training

## Visualizations

The notebook includes visualizations for:
- Positional encoding patterns
- Attention weight heatmaps
- Training metrics (loss and accuracy)

## Key Implementation Details

- **Scaled Dot-Product Attention**: `attention = softmax(QK^T / sqrt(d_k))V`
- **Causal Masking**: Prevents decoder from attending to future positions
- **Layer Normalization**: Applied after residual connections
- **Dropout**: Configurable dropout for regularization

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper by Vaswani et al. (2017)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar's visual guide
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## License

This project is open source and available for educational purposes.

## Author

**aaishwarymishra**

If you find this implementation helpful, please consider giving it a star!

---

*Built with TensorFlow and inspired by the revolutionary Transformer architecture*
