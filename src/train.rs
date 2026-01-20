use crate::data::{TextBatch, TextBatcher, TextDataset};
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainStep,
        metric::LossMetric,
    },
};
use std::{fmt::Display, fs};

/// Training loop for language models.
pub fn lm_train<B, M>(
    artifact_dir: &str,
    model: M,
    num_epochs: usize,
    batch_size: usize,
    num_workers: usize,
    seed: u64,
    learning_rate: f64,
    device: &B::Device,
) where
    B: AutodiffBackend,
    M: TrainStep<Input = TextBatch<B>, Output = ClassificationOutput<B>>
        + AutodiffModule<B>
        + Display
        + 'static,
    M::InnerModule: InferenceStep<
            Input = TextBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    B::seed(device, seed);

    let text = fs::read_to_string("data/input.txt").unwrap();
    let dataset = TextDataset::new(&text);

    let dataloader_train = DataLoaderBuilder::new(TextBatcher::new())
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset.train());
    let dataloader_valid = DataLoaderBuilder::new(TextBatcher::new())
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset.valid());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(num_epochs)
        .summary();

    let result = training.launch(Learner::new(
        model,
        AdamWConfig::new().init(),
        learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully")
}
