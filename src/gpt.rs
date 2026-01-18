use crate::data::{N, TextBatch, TextBatcher, TextDataset};
use crate::infer::LanguageModelInference;
use crate::utils::{create_artifact_dir, multinomial_sample};
use burn::nn::{Linear, LinearConfig};
use burn::{
    Tensor,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig},
    optim::AdamWConfig,
    prelude::{Backend, Int},
    record::CompactRecorder,
    tensor::{Bool, TensorData, activation::softmax, backend::AutodiffBackend, s},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::LossMetric,
    },
};
use std::fs;

#[derive(Debug, Module)]
pub struct Head<B: Backend> {
    key: Linear<B>,
    query: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> Head<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [b, t, c] = x.shape().dims();

        let k = self.key.forward(x.clone());
        let q = self.query.forward(x.clone());
        let v = self.value.forward(x);

        let mask = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &device).expand([b, t, t]);

        // compute attention scores ("affinities")
        let wei = q.matmul(k.t()) / (c as f32).sqrt();
        let wei = wei.mask_fill(mask, f32::MIN);
        let wei = softmax(wei, 2);

        // perform the weighted aggregation of the values
        wei.matmul(v)
    }
}

#[derive(Debug, Config)]
pub struct HeadConfig {
    pub n_embd: usize,
    pub head_size: usize,
}

impl HeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Head<B> {
        Head {
            key: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
            query: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct NanoGPTModel<B: Backend> {
    token_embedding_table: Embedding<B>,
    position_embedding_table: Embedding<B>,
    sa_head: Head<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> NanoGPTModel<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = idx.shape().dims();
        let device = idx.device();

        let ts = Tensor::arange(0..(t as i64), &device).expand([b, t]);
        let tok_emb = self.token_embedding_table.forward(idx);
        let pos_emb = self.position_embedding_table.forward(ts);
        let x = tok_emb + pos_emb;
        let x = self.sa_head.forward(x);
        let logits = self.lm_head.forward(x);

        logits
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

impl<B: Backend> LanguageModelInference for NanoGPTModel<B> {
    type Tokens = Tensor<B, 2, Int>;

    fn generate(&self, idx: Self::Tokens, max_new_tokens: usize) -> Self::Tokens {
        let mut idx = idx;
        let device = idx.device();

        for _ in 0..max_new_tokens {
            // get the preditctions
            let x = idx.clone().slice(s![.., -(N as i32)..]);
            let logits = self.forward(x);
            // focus only on the last time step
            let logits = logits.slice(s![.., -1, ..]).reshape([0, -1]); // becomes (B, C)
            // apply softmax to get probabilities
            let probs = softmax(logits, 1);
            // sample from the distribution
            let idx_next = probs
                .iter_dim(0)
                .map(|p| {
                    p.reshape([-1])
                        .into_data()
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

impl<B: AutodiffBackend> TrainStep for NanoGPTModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.forward_classification(item.context, item.targets);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> InferenceStep for NanoGPTModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item.context, item.targets)
    }
}

#[derive(Debug, Config)]
pub struct NanoGPTModelConfig {
    pub vocabulary: Vec<u8>,
    pub n_embd: usize,
    pub head_size: usize,
}

impl NanoGPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NanoGPTModel<B> {
        let vocab_size = self.vocabulary.len();

        NanoGPTModel {
            token_embedding_table: EmbeddingConfig::new(vocab_size, self.n_embd).init(device),
            position_embedding_table: EmbeddingConfig::new(N, self.n_embd).init(device),
            sa_head: HeadConfig::new(self.n_embd, self.head_size).init(device),
            lm_head: LinearConfig::new(self.n_embd, vocab_size).init(device),
        }
    }
}

#[derive(Config, Debug)]
pub struct NanoGPTModelTrainingConfig {
    pub model: NanoGPTModelConfig,
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

pub fn nanogpt_train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: NanoGPTModelTrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(device, config.seed);

    let text = fs::read_to_string("data/input.txt").unwrap();
    let dataset = TextDataset::new(&text);

    let dataloader_train = DataLoaderBuilder::new(TextBatcher::new())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(dataset.train());
    let dataloader_valid = DataLoaderBuilder::new(TextBatcher::new())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(dataset.valid());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    // Initialize the model on the autodiff backend `B` (so Training can compute gradients).
    let model = config.model.init::<B>(device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully")
}

#[cfg(test)]
mod tests {
    use super::*;

    type MyBackend = burn::backend::Wgpu<f32, i32>;

    #[test]
    fn test_head_output_shape() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let dist = burn::tensor::Distribution::Normal(0.0, 1.0);
        let x = Tensor::<MyBackend, 3>::random([4, 8, 32], dist, &device);
        let head = HeadConfig::new(32, 16).init(&device);
        let out = head.forward(x);

        assert_eq!(out.shape().dims(), [4, 8, 16]);
    }
}
