use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use std::fs;

pub fn multinomial_sample(probs: &[f64], num_samples: usize) -> Vec<usize> {
    assert!(!probs.is_empty(), "probs must not be empty");
    // probs must be non-negative and not all zero
    let dist = WeightedIndex::new(probs).expect("invalid probabilities");
    let mut rng = rand::rng();

    (0..num_samples).map(|_| dist.sample(&mut rng)).collect()
}

pub fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}
