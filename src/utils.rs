use burn::{
    Tensor,
    config::Config,
    module::{AutodiffModule, Module},
    prelude::{Backend, Int},
    record::Record,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainStep},
};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use std::{fmt::Display, fs};

use crate::{
    data::TextBatch,
    infer::LanguageModelInference,
};

#[derive(Debug, PartialEq)]
pub enum LanguageModel {
    Bigram,
    NanoGPT,
}

impl TryFrom<String> for LanguageModel {
    type Error = pico_args::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "bigram" => Ok(LanguageModel::Bigram),
            "nanogpt" => Ok(LanguageModel::NanoGPT),
            _ => Err(pico_args::Error::ArgumentParsingFailed {
                cause: format!("Unknown model: {}; possible values: bigram, nanogpt", value),
            }),
        }
    }
}

#[derive(Debug)]
pub struct AppArgs {
    pub model: LanguageModel,
    pub train: bool,
    pub input: String,
    pub num_epochs: usize,
    pub num_workers: usize,
    pub artifact_dir: String,
    pub infer: bool,
    pub context: String,
    pub max_tokens: usize,
}

pub fn parse_args() -> Result<AppArgs, pico_args::Error> {
    let mut pargs = pico_args::Arguments::from_env();

    let args = AppArgs {
        model: LanguageModel::try_from(
            pargs
                .value_from_str::<&str, String>("--model")
                .unwrap_or_default(),
        )?,
        train: pargs.contains("--train"),
        input: pargs
            .value_from_str(["-i", "--input"])
            .unwrap_or("data/input.txt".to_string()),
        num_epochs: pargs.value_from_str(["-n", "--n-epochs"]).unwrap_or(5),
        num_workers: pargs.value_from_str(["-p", "--n-workers"]).unwrap_or(4),
        artifact_dir: pargs
            .value_from_str(["-o", "--output"])
            .unwrap_or("bigram".to_string()),
        infer: pargs.contains("--infer"),
        context: pargs
            .value_from_str(["-c", "--context"])
            .unwrap_or_default(),
        max_tokens: pargs.value_from_str(["-m", "--max-tokens"]).unwrap_or(100),
    };
    _ = pargs.finish();

    Ok(args)
}

pub fn multinomial_sample(probs: &[f64], num_samples: usize) -> Vec<usize> {
    assert!(!probs.is_empty(), "probs must not be empty");
    // probs must be non-negative and not all zero
    let dist = WeightedIndex::new(probs).expect("invalid probabilities");
    let mut rng = rand::rng();

    (0..num_samples).map(|_| dist.sample(&mut rng)).collect()
}

pub fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

pub trait ModelSpec<B, AD>
where
    B: Backend,
    AD: AutodiffBackend<InnerBackend = B>,
{
    type TrainConfig: Config;
    type TrainModel: TrainStep<Input = TextBatch<AD>, Output = ClassificationOutput<AD>>
        + AutodiffModule<AD>
        + Display
        + 'static;
    type InferModel: Module<B, Record = Self::Record>
        + LanguageModelInference<Tokens = Tensor<B, 2, Int>>;
    type Record: Record<B>;

    fn build_config(vocab: Vec<u8>, args: &AppArgs) -> Self::TrainConfig;
    fn train_params(config: &Self::TrainConfig) -> (usize, usize, usize, u64, f64);
    fn init_train(config: &Self::TrainConfig, device: &AD::Device) -> Self::TrainModel;
    fn init_infer(config: &Self::TrainConfig, device: &B::Device) -> Self::InferModel;
}
