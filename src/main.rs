pub mod train;
pub mod utils;

use crate::train::{BigramLanguageModelConfig, TextBatcher, TextDataset, Tokenizer};
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataloader::DataLoaderBuilder;
use burn::prelude::*;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let text = fs::read_to_string("data/input.txt")?;
    let dataset = TextDataset::new(&text);
    let train = dataset.train();

    let batcher = TextBatcher::<MyBackend>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .build(train);

    // Test how data looks
    for (it, batch) in dataloader.iter().take(2).enumerate() {
        println!(
            "[Iteration: {}] Context: {} | Targets: {}",
            it, batch.context, batch.targets
        );
    }

    let batch = dataloader.iter().next().unwrap();
    let (xb, yb) = (batch.context, batch.targets);

    // Bigram model
    let vocab_size = dataset.vocabulary.size();
    let bigram_model = BigramLanguageModelConfig::new(vocab_size).init::<MyBackend>(&device);
    let out = bigram_model.forward(xb.clone());
    println!("{:?}", out.shape());
    let cls = bigram_model.forward_classification(xb, yb);
    println!("Loss: {}", cls.loss.to_data());

    // Generate from the model
    let idx = Tensor::<MyBackend, 2, Int>::zeros(Shape::new([1, 1]), &device);
    let output = bigram_model.generate(idx, 100);
    let tokens: Vec<u8> = output
        .slice(0)
        .into_data()
        .iter()
        .map(|x: i32| x as u8)
        .collect();
    let sentence = dataset.vocabulary.decode(&tokens);
    println!("Generated: {sentence}");

    // Training

    Ok(())
}
