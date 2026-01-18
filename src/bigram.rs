use crate::data::{TextBatch, TextBatcher, TextDataset};
use crate::infer::LanguageModelInference;
use crate::utils::{create_artifact_dir, multinomial_sample};
use burn::{
    Tensor,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLossConfig},
    optim::AdamWConfig,
    prelude::{Backend, Int},
    record::CompactRecorder,
    tensor::{TensorData, activation::softmax, backend::AutodiffBackend, s},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::LossMetric,
    },
};
use std::fs;

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

impl<B: Backend> LanguageModelInference for BigramLanguageModel<B> {
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
    pub n_embd: usize,
    pub head_size: usize,
}

impl BigramLanguageModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramLanguageModel<B> {
        let vocab_size = self.vocabulary.len();

        BigramLanguageModel {
            token_embedding_table: EmbeddingConfig::new(vocab_size, self.n_embd).init(device),
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

pub fn bigram_lm_train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: BigramLanguageModelTrainingConfig,
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
