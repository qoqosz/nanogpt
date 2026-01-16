use crate::data::TextBatch;
use crate::utils::multinomial_sample;
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig},
    prelude::{Backend, Int},
    tensor::{TensorData, activation::softmax, backend::AutodiffBackend, s},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

pub trait LanguageModel {
    type Tokens;

    fn generate(&self, idx: Self::Tokens, max_new_tokens: usize) -> Self::Tokens;
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
}

// TODO: try to make it more generic
impl<B: Backend> LanguageModel for BigramLanguageModel<B> {
    type Tokens = Tensor<B, 2, Int>;

    fn generate(&self, idx: Self::Tokens, max_new_tokens: usize) -> Self::Tokens {
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

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.forward_classification(item.context, item.targets);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> InferenceStep for BigramLanguageModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item.context, item.targets)
    }
}

#[derive(Debug, Config)]
pub struct BigramLanguageModelConfig {
    pub vocabulary: Vec<u8>,
}

impl BigramLanguageModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramLanguageModel<B> {
        BigramLanguageModel {
            token_embedding_table: EmbeddingConfig::new(
                self.vocabulary.len(),
                self.vocabulary.len(),
            )
            .init(device),
        }
    }
}
