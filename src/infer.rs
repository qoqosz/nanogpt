use crate::bigram::BigramLanguageModelTrainingConfig;
use crate::data::{Tokenizer, Vocabulary};
use crate::gpt::NanoGPTModelTrainingConfig;
use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{Backend, Int},
    record::{CompactRecorder, Recorder},
    tensor::TensorData,
};

pub trait LanguageModelInference {
    type Tokens;

    fn generate(&self, idx: Self::Tokens, max_new_tokens: usize) -> Self::Tokens;
}

pub fn bigram_lm_infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    context: &str,
    max_new_tokens: usize,
) {
    let config = BigramLanguageModelTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init(device).load_record(record);

    let vocabulary = Vocabulary::new(&config.model.vocabulary);
    let data = TensorData::from(vocabulary.encode(context).as_slice());
    let idx = Tensor::<B, 1, Int>::from_data(data, device).reshape([1, -1]);
    let output = model.generate(idx, max_new_tokens);
    let tokens: Vec<u8> = output
        .slice(0)
        .into_data()
        .iter()
        .map(|x: i32| x as u8)
        .collect();

    println!("Predicted: {}", vocabulary.decode(&tokens));
}

pub fn nanogpt_infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    context: &str,
    max_new_tokens: usize,
) {
    let config = NanoGPTModelTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init(device).load_record(record);

    let vocabulary = Vocabulary::new(&config.model.vocabulary);
    let data = TensorData::from(vocabulary.encode(context).as_slice());
    let idx = Tensor::<B, 1, Int>::from_data(data, device).reshape([1, -1]);
    let output = model.generate(idx, max_new_tokens);
    let tokens: Vec<u8> = output
        .slice(0)
        .into_data()
        .iter()
        .map(|x: i32| x as u8)
        .collect();

    println!("Predicted: {}", vocabulary.decode(&tokens));
}
