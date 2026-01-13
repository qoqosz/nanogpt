use burn::{
    Tensor,
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::{Backend, Int},
    tensor::TensorData,
};
use rustc_hash::{FxHashMap, FxHashSet};

static N: usize = 8;

pub trait Tokenizer {
    // Associated token type
    type Token;

    // Encode a string to a vector of tokens.
    fn encode(&self, txt: &str) -> Vec<Self::Token>;

    // Decode a vector of tokens to a string.
    fn decode(&self, tokens: &[Self::Token]) -> String;
}

#[derive(Clone)]
pub struct Vocabulary {
    chars: FxHashSet<u8>,
    stoi: FxHashMap<u8, u8>,
    itos: FxHashMap<u8, u8>,
}

impl Vocabulary {
    pub fn new(chars: &[u8]) -> Self {
        let chars: FxHashSet<u8> = chars.iter().copied().collect();
        let stoi: FxHashMap<u8, u8> = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i as u8))
            .collect();
        let itos: FxHashMap<u8, u8> = stoi.iter().map(|(ch, i)| (*i, *ch)).collect();

        Self { chars, stoi, itos }
    }

    pub fn size(&self) -> usize {
        self.chars.len()
    }
}

impl<T> From<T> for Vocabulary
where
    T: AsRef<str>,
{
    fn from(value: T) -> Self {
        Vocabulary::new(value.as_ref().as_bytes())
    }
}

impl Tokenizer for Vocabulary {
    type Token = u8;

    fn encode(&self, txt: &str) -> Vec<Self::Token> {
        txt.as_bytes()
            .iter()
            .flat_map(|ch| self.stoi.get(ch))
            .copied()
            .collect()
    }

    fn decode(&self, tokens: &[Self::Token]) -> String {
        let bytes = tokens
            .iter()
            .flat_map(|i| self.itos.get(i))
            .copied()
            .collect::<Vec<_>>();
        std::str::from_utf8(&bytes).unwrap().to_owned()
    }
}

#[derive(Debug, Clone)]
pub struct TextItem {
    pub x: [u8; 8],
    pub y: [u8; 8],
}

pub struct TextDataset {
    vocabulary: Vocabulary,
    tokens: Vec<u8>,
}

impl TextDataset {
    pub fn new(txt: &str) -> Self {
        let vocabulary = Vocabulary::from(&txt);
        let tokens = vocabulary.encode(&txt);

        Self { vocabulary, tokens }
    }

    pub fn train(&self) -> Self {
        let n = (self.tokens.len() as f64 * 0.9) as usize;
        Self {
            vocabulary: self.vocabulary.clone(),
            tokens: self.tokens[..n].to_vec(),
        }
    }

    pub fn test(&self) -> Self {
        let n = (self.tokens.len() as f64 * 0.9) as usize;
        Self {
            vocabulary: self.vocabulary.clone(),
            tokens: self.tokens[n..].to_vec(),
        }
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        if let Some(v) = self.tokens.get(index * (N + 1)..index * (N + 1) + N + 1) {
            Some(TextItem {
                x: v[..N].try_into().unwrap(),
                y: v[1..].try_into().unwrap(),
            })
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.tokens.len() / (N + 1)
    }
}

#[allow(unused)]
#[derive(Clone)]
pub struct TextBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub context: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        let context = items
            .iter()
            .map(|item| TensorData::from(item.x))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, N]))
            .collect();
        let targets = items
            .iter()
            .map(|item| TensorData::from(item.y))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, N]))
            .collect();

        let context = Tensor::cat(context, 0);
        let targets = Tensor::cat(targets, 0);

        TextBatch { context, targets }
    }
}

#[derive(Debug, Module)]
pub struct BigramLanguageModel<B: Backend> {
    token_embedding_table: Embedding<B>,
}

impl<B: Backend> BigramLanguageModel<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding_table.forward(idx)
    }
}

#[derive(Debug, Config)]
pub struct BigramLanguageModelConfig {
    vocab_size: usize,
}

impl BigramLanguageModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramLanguageModel<B> {
        BigramLanguageModel {
            token_embedding_table: EmbeddingConfig::new(self.vocab_size, self.vocab_size)
                .init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let input = "hii there";
        let vocab = Vocabulary::from(input);
        assert_eq!(input, vocab.decode(&vocab.encode(input)));
    }
}
