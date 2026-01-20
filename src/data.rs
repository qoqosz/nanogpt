use burn::{
    Tensor,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::{Backend, Int},
    serde::{Deserialize, Serialize},
    tensor::TensorData,
};
use itertools::Itertools;
use std::collections::HashMap;
use std::marker::PhantomData;

// Context window size
pub static N: usize = 32;

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
    // Actual vocabulary
    pub chars: Vec<u8>,
    // Char to int mapping
    stoi: HashMap<u8, u8>,
    // Int to char mapping
    itos: HashMap<u8, u8>,
}

impl Vocabulary {
    pub fn new(chars: &[u8]) -> Self {
        let chars: Vec<u8> = chars.iter().copied().unique().sorted_unstable().collect();
        let stoi: HashMap<u8, u8> = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i as u8))
            .collect();
        let itos: HashMap<u8, u8> = stoi.iter().map(|(ch, i)| (*i, *ch)).collect();

        Self { chars, stoi, itos }
    }

    // Number of unique chars in a vocabulary
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextItem {
    pub x: [u8; N],
    pub y: [u8; N],
}

pub struct TextDataset {
    pub vocabulary: Vocabulary,
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

    pub fn valid(&self) -> Self {
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
    _b: PhantomData<B>,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new() -> Self {
        Self { _b: PhantomData }
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
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();
        let targets = items
            .iter()
            .map(|item| TensorData::from(item.y))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let context = Tensor::cat(context, 0);
        let targets = Tensor::cat(targets, 0);

        TextBatch { context, targets }
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
