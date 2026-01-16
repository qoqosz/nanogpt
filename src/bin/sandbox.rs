use burn::{backend::Wgpu, data::dataloader::DataLoaderBuilder};
use nanogpt::data::{TextBatcher, TextDataset};
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32, i32>;

    let text = fs::read_to_string("data/input.txt")?;
    let dataset = TextDataset::new(&text);
    let train = dataset.train();

    let batcher = TextBatcher::<MyBackend>::new();
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

    Ok(())
}
