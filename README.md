# nanogpt

A small, educational Rust implementation inspired by Andrej Karpathy's "[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)" video. This repository contains a compact from-scratch GPT-style implementation built using the Rust ecosystem and the burn crate.

## Why this repository

- Educational: reimplements the core ideas from Karpathy's walkthrough in Rust to illustrate model structure, training loop, and minimal tooling.
- Minimal: focuses on a small, readable codebase rather than production-scale features.
- Hands-on: two example binaries you can build and run locally.

## Quickstart

### Prerequisites
- Rust toolchain (stable) with Cargo installed. Install from https://www.rust-lang.org/tools/install.

### Build

Run the default build:

```
cargo build --release
```

#### Run the examples

Run the bigram example:

```
cargo run --bin bigram -- --train  -n 1 -p 10 -o /tmp/bigram/ --infer -c "What time is it?" -m 50
```

## Notes

- This repository is intended as an educational reference and a hands-on port; it is not optimized for large-scale training or production use.
- Model hyperparameters, dataset handling, and training routines are intentionally minimal to keep the code easy to follow.

## References

- Andrej Karpathy — "Let's build GPT": https://www.youtube.com/watch?v=kCc8FmEb1nY and https://github.com/karpathy/nanoGPT
