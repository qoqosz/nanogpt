#![recursion_limit = "256"]
pub mod bigram;
pub mod data;
pub mod gpt;
pub mod infer;
pub mod train;
pub mod utils;

use burn::{
    Tensor,
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    module::{AutodiffModule, Module},
    prelude::Int,
    record::{CompactRecorder, Record, Recorder},
    train::{ClassificationOutput, InferenceStep, TrainStep},
};
use nanogpt::{
    bigram::BigramSpec,
    data::{TextBatch, TextDataset},
    gpt::NanoGPTSpec,
    infer::{LanguageModelInference, lm_infer},
    train::lm_train,
    utils::{AppArgs, LanguageModel, ModelSpec, create_artifact_dir, parse_args},
};
use std::{fmt::Display, fs};

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn run_model<S>(args: &AppArgs, dataset: &TextDataset, device: &WgpuDevice)
where
    S: ModelSpec<MyBackend, MyAutodiffBackend>,
    S::TrainModel: TrainStep<
            Input = TextBatch<MyAutodiffBackend>,
            Output = ClassificationOutput<MyAutodiffBackend>,
        > + AutodiffModule<MyAutodiffBackend>
        + Display
        + 'static,
    <S::TrainModel as AutodiffModule<MyAutodiffBackend>>::InnerModule:
        InferenceStep<Input = TextBatch<MyBackend>, Output = ClassificationOutput<MyBackend>>,
    S::InferModel: Module<MyBackend, Record = S::Record>
        + LanguageModelInference<Tokens = Tensor<MyBackend, 2, Int>>,
    S::Record: Record<MyBackend>,
{
    let config = S::build_config(dataset.vocabulary.chars.clone(), args);

    if args.train {
        create_artifact_dir(&args.artifact_dir);
        config
            .save(format!("{}/config.json", args.artifact_dir))
            .expect("Config should be saved successfully");

        let (num_epochs, batch_size, num_workers, seed, learning_rate) = S::train_params(&config);
        lm_train::<MyAutodiffBackend, S::TrainModel>(
            &args.artifact_dir,
            S::init_train(&config, device),
            num_epochs,
            batch_size,
            num_workers,
            seed,
            learning_rate,
            device,
        );
    }

    if args.infer {
        let record: S::Record = CompactRecorder::new()
            .load(format!("{}/model", args.artifact_dir).into(), device)
            .expect("Trained model should exist; run train first");
        let model = S::init_infer(&config, device).load_record(record);
        let predicted = lm_infer(
            model,
            dataset.vocabulary.clone(),
            &args.context,
            args.max_tokens,
            device,
        );
        println!("Predicted: {predicted}");
    }
}

fn main() -> Result<(), pico_args::Error> {
    let args = parse_args()?;

    let device = WgpuDevice::default();
    let text = fs::read_to_string(&args.input).expect("Input file does not exist");
    let dataset = TextDataset::new(&text);

    println!("Selected model: {:?}", args.model);

    match args.model {
        LanguageModel::Bigram => run_model::<BigramSpec>(&args, &dataset, &device),
        LanguageModel::NanoGPT => run_model::<NanoGPTSpec>(&args, &dataset, &device),
    }

    Ok(())
}
