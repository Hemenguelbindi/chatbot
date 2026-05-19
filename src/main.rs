mod data;
mod model;
mod tokenizer;
mod training;

use burn::backend::{Autodiff, Flex};

type MyBackend = Autodiff<Flex>;

fn main() {
    // Use Flex backend (CPU, JIT-compiled — faster than NdArray)
    let device = Default::default();
    training::run::<MyBackend>(device);
}
