#![recursion_limit = "256"]
pub mod bigram;
pub mod data;
pub mod gpt;
pub mod infer;
pub mod utils;

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    optim::AdamWConfig,
};
use nanogpt::{
    bigram::{BigramLanguageModelConfig, BigramLanguageModelTrainingConfig, bigram_lm_train},
    data::TextDataset,
    gpt::{NanoGPTModelConfig, NanoGPTModelTrainingConfig, nanogpt_train},
    infer::{bigram_lm_infer, nanogpt_infer},
};
use std::fs;

#[derive(Debug)]
struct AppArgs {
    model: String,
    train: bool,
    input: String,
    num_epochs: usize,
    num_workers: usize,
    artifact_dir: String,
    infer: bool,
    context: String,
    max_tokens: usize,
}

fn parse_args() -> Result<AppArgs, pico_args::Error> {
    let mut pargs = pico_args::Arguments::from_env();

    let args = AppArgs {
        model: pargs
            .value_from_str("--model")
            .unwrap_or("bigram".to_string()),
        train: pargs.contains(["-t", "--train"]),
        input: pargs
            .value_from_str("-i")
            .unwrap_or("data/input.txt".to_string()),
        num_epochs: pargs.value_from_str("-n").unwrap_or(5),
        num_workers: pargs.value_from_str("-p").unwrap_or(4),
        artifact_dir: pargs.value_from_str("-o").unwrap_or("bigram".to_string()),
        infer: pargs.contains(["-j", "--infer"]),
        context: pargs.value_from_str("-c").unwrap_or_default(),
        max_tokens: pargs.value_from_str("-m").unwrap_or(100),
    };
    _ = pargs.finish();

    Ok(args)
}

fn main() -> Result<(), pico_args::Error> {
    let args = parse_args()?;

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();

    let text = fs::read_to_string(args.input).expect("Input file does not exist");
    let dataset = TextDataset::new(&text);

    println!("Selected model: {}", args.model);

    if args.model == "bigram".to_string() {
        // Training
        if args.train {
            let config = BigramLanguageModelTrainingConfig::new(
                BigramLanguageModelConfig::new(dataset.vocabulary.chars, 32, 16),
                AdamWConfig::new(),
            )
            .with_num_epochs(args.num_epochs)
            .with_num_workers(args.num_workers);
            bigram_lm_train::<MyAutodiffBackend>(&args.artifact_dir, config, &device);
        }

        // Inference
        if args.infer {
            bigram_lm_infer::<MyBackend>(
                &args.artifact_dir,
                &device,
                &args.context,
                args.max_tokens,
            );
        }
    } else {
        // Training
        if args.train {
            let config = NanoGPTModelTrainingConfig::new(
                NanoGPTModelConfig::new(dataset.vocabulary.chars, 32, 4, 32, 3),
                AdamWConfig::new(),
            )
            .with_num_epochs(args.num_epochs)
            .with_num_workers(args.num_workers);
            nanogpt_train::<MyAutodiffBackend>(&args.artifact_dir, config, &device);
        }

        // Inference
        if args.infer {
            nanogpt_infer::<MyBackend>(&args.artifact_dir, &device, &args.context, args.max_tokens);
        }
    }

    Ok(())
}
