use crate::utils::multinomial_sample;
use burn::record::CompactRecorder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};
use burn::{
    Tensor,
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::Module,
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig},
    optim::AdamWConfig,
    prelude::{Backend, Int},
    tensor::{TensorData, activation::softmax, backend::AutodiffBackend, s},
    train::{ClassificationOutput, TrainOutput, TrainStep},
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::fs;

// Context window size
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
    // Actual vocabulary
    chars: FxHashSet<u8>,
    // Char to int mapping
    stoi: FxHashMap<u8, u8>,
    // Int to char mapping
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
    pub x: [u8; 8],
    pub y: [u8; 8],
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

    pub fn forward_classification(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(idx);
        let [b, t, c] = output.dims();
        let output = output.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }

    pub fn generate(&self, idx: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut idx = idx;
        let device = idx.device();

        for _ in 0..max_new_tokens {
            // get the preditctions
            let logits = self.forward(idx.clone());
            // focus only on the last time step
            let logits = logits.slice(s![.., -1, ..]).reshape([0, -1]); // becomes (B, C)
            // apply softmax to get probabilities
            let probs = softmax(logits, 1);
            // sample from the distribution
            let idx_next = probs
                .iter_dim(0)
                .map(|p| p.reshape([-1]))
                .map(|p| {
                    p.into_data()
                        .iter()
                        .map(|x: f32| x as f64)
                        .collect::<Vec<f64>>()
                })
                .map(|p| multinomial_sample(&p, 1))
                .map(|x| TensorData::from(x.as_slice()))
                .map(|data| Tensor::<B, 1, Int>::from_data(data, &device))
                .map(|tensor| tensor.reshape([1, -1]))
                .collect();
            let idx_next = Tensor::cat(idx_next, 0);
            idx = Tensor::cat(vec![idx, idx_next], 1);
        }

        idx
    }
}

impl<B: AutodiffBackend> TrainStep for BigramLanguageModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_classification(item.context, item.targets);

        TrainOutput::new(self, output.loss.backward(), output)
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

#[derive(Config, Debug)]
pub struct BigramLanguageModelTrainingConfig {
    pub model: BigramLanguageModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 6)]
    pub num_workers: usize,
    #[config(default = 43)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

pub fn bigram_lm_train<B: AutodiffBackend<InnerBackend = B>>(
    artifact_dir: &str,
    config: BigramLanguageModelTrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let text = fs::read_to_string("data/input.txt").unwrap();
    let dataset = TextDataset::new(&text);

    let batcher = TextBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(dataset.train());
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(dataset.test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));
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
