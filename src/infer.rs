use crate::data::Tokenizer;
use burn::{
    Tensor,
    prelude::{Backend, Int},
    tensor::TensorData,
};

pub trait LanguageModelInference {
    type Tokens;

    fn generate(&self, idx: Self::Tokens, max_new_tokens: usize) -> Self::Tokens;
}

pub fn lm_infer<B, M, T>(
    model: M,
    tokenizer: T,
    context: &str,
    max_tokens: usize,
    device: &B::Device,
) -> String
where
    B: Backend,
    M: LanguageModelInference<Tokens = Tensor<B, 2, Int>>,
    T: Tokenizer<Token = u8>,
{
    let data = TensorData::from(tokenizer.encode(context).as_slice());
    let idx = Tensor::<B, 1, Int>::from_data(data, device).reshape([1, -1]);
    let output = model.generate(idx, max_tokens);
    let tokens: Vec<u8> = output
        .slice(0)
        .into_data()
        .iter()
        .map(|x: i32| x as u8)
        .collect();

    tokenizer.decode(&tokens)
}
