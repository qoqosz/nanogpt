#![recursion_limit = "256"]
pub mod bigram;
pub mod data;
pub mod gpt;
pub mod infer;
pub mod train;
pub mod utils;

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
};
use nanogpt::{
    bigram::{
        BigramLanguageModel, BigramLanguageModelConfig, BigramLanguageModelRecord,
        BigramLanguageModelTrainingConfig,
    },
    data::TextDataset,
    gpt::{NanoGPTModel, NanoGPTModelConfig, NanoGPTModelRecord, NanoGPTModelTrainingConfig},
    infer::lm_infer,
    train::lm_train,
    utils::{LanguageModel, create_artifact_dir, parse_args},
};
use std::fs;

fn main() -> Result<(), pico_args::Error> {
    let args = parse_args()?;

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();
    let text = fs::read_to_string(args.input).expect("Input file does not exist");
    let dataset = TextDataset::new(&text);

    println!("Selected model: {:?}", args.model);

    match args.model {
        LanguageModel::Bigram => {
            let config = BigramLanguageModelTrainingConfig::new(BigramLanguageModelConfig::new(
                dataset.vocabulary.chars.clone(),
                32,
                16,
            ))
            .with_num_epochs(args.num_epochs)
            .with_num_workers(args.num_workers);

            if args.train {
                create_artifact_dir(&args.artifact_dir);
                config
                    .save(format!("{}/config.json", args.artifact_dir))
                    .expect("Config should be saved successfully");
                lm_train::<MyAutodiffBackend, BigramLanguageModel<MyAutodiffBackend>>(
                    &args.artifact_dir,
                    config.model.init(&device),
                    config.num_epochs,
                    config.batch_size,
                    config.num_workers,
                    config.seed,
                    config.learning_rate,
                    &device,
                );
            }

            if args.infer {
                let record: BigramLanguageModelRecord<MyBackend> = CompactRecorder::new()
                    .load(format!("{}/model", args.artifact_dir).into(), &device)
                    .expect("Trained model should exist; run train first");
                let model = config.model.init(&device).load_record(record);
                let predicted = lm_infer(
                    model,
                    dataset.vocabulary,
                    &args.context,
                    args.max_tokens,
                    &device,
                );
                println!("Predicted: {predicted}");
            }
        }
        LanguageModel::NanoGPT => {
            let config = NanoGPTModelTrainingConfig::new(NanoGPTModelConfig::new(
                dataset.vocabulary.chars.clone(),
                32,
                4,
                32,
                3,
            ))
            .with_num_epochs(args.num_epochs)
            .with_num_workers(args.num_workers);

            if args.train {
                create_artifact_dir(&args.artifact_dir);
                config
                    .save(format!("{}/config.json", args.artifact_dir))
                    .expect("Config should be saved successfully");
                lm_train::<MyAutodiffBackend, NanoGPTModel<MyAutodiffBackend>>(
                    &args.artifact_dir,
                    config.model.init(&device),
                    config.num_epochs,
                    config.batch_size,
                    config.num_workers,
                    config.seed,
                    config.learning_rate,
                    &device,
                );
            }

            if args.infer {
                let record: NanoGPTModelRecord<MyBackend> = CompactRecorder::new()
                    .load(format!("{}/model", args.artifact_dir).into(), &device)
                    .expect("Trained model should exist; run train first");
                let model = config.model.init(&device).load_record(record);
                let predicted = lm_infer(
                    model,
                    dataset.vocabulary,
                    &args.context,
                    args.max_tokens,
                    &device,
                );
                println!("Predicted: {predicted}");
            }
        }
    }

    Ok(())
}
