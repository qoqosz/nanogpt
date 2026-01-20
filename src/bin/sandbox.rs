#![allow(unused, unused_imports)]
use burn::Tensor;
use burn::backend::wgpu::WgpuDevice;
use burn::nn::LinearConfig;
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

    // Test tensors that contribute to the head network
    let (b, t, c) = (4, 8, 32);
    let distribution = burn::tensor::Distribution::Normal(0.0, 1.0);
    let x = Tensor::<MyBackend, 3>::random([b, t, c], distribution, &device);

    println!("{x}");

    let head_size = 16;

    let key = LinearConfig::new(c, head_size)
        .with_bias(false)
        .init(&device);
    let query = LinearConfig::new(c, head_size)
        .with_bias(false)
        .init(&device);
    let value = LinearConfig::new(c, head_size)
        .with_bias(false)
        .init(&device);

    let k = key.forward(x.clone());
    let q = query.forward(x.clone());
    let v = value.forward(x);

    let wei = q.matmul(k.t());

    println!("wei: {wei}");

    let block_size = 32;
    let mask =
        Tensor::<MyBackend, 2, Bool>::tril_mask([t.min(block_size), t.min(block_size)], 0, &device);
    println!("mask: {mask}");

    let mask = mask.expand([b, t, t]);
    let wei = wei.mask_fill(mask, f32::MIN);
    println!("wei: {wei}");

    let wei = softmax(wei, 2);
    println!("wei: {wei}");

    let out = wei.matmul(v);

    println!("out: {out}");

    // Embeddings test
    let lm_head = LinearConfig::new(32, 65).init::<MyBackend>(&device);
    let x = Tensor::<MyBackend, 3>::random([16, 8, 32], distribution, &device);

    Ok(())
}
