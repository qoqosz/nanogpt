use crate::data::{TextBatcher, TextDataset};
use crate::model::BigramLanguageModelConfig;
use crate::utils::create_artifact_dir;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};
use std::fs;

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
