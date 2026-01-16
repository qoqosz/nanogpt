use burn::Tensor;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Bool;
use burn::tensor::activation::softmax;
use burn::{backend::Wgpu, data::dataloader::DataLoaderBuilder};
use nanogpt::data::{TextBatcher, TextDataset};
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32, i32>;

    let device = WgpuDevice::default();

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

    // the trick in self attention
    let wei = Tensor::<MyBackend, 2>::zeros([8, 8], &device);
    let mask = Tensor::<MyBackend, 2, Bool>::tril_mask([8, 8], 0, &device);
    let wei = wei.mask_fill(mask, f32::MIN);
    let wei = softmax(wei, 1);
    println!("{}", wei);

    Ok(())
}
