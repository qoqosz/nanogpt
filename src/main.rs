pub mod train;

use crate::train::{TextBatcher, TextDataset};
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataloader::DataLoaderBuilder;
// use burn::prelude::*;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let text = fs::read_to_string("data/input.txt")?;
    let dataset = TextDataset::new(&text);
    let train = dataset.train();

    let batcher = TextBatcher::<MyBackend>::new(device);
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .build(train);

    for (it, batch) in dataloader.iter().take(2).enumerate() {
        println!(
            "[Iteration: {}] Context: {} | Targets: {}",
            it, batch.context, batch.targets
        );
    }

    Ok(())
}
