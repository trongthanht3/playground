use crate::lib::layer::Layer;
use core::fmt;
use ndarray::prelude::*;
use std::collections::HashMap;

#[derive(Default, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
    pub weight: HashMap<String, Array2<f32>>,
}

impl NeuralNetwork {
    // Init
    // pub fn init_params(&self) -> HashMap<String, Array2<f32>> {
    // }

    // Add layers
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNetwork")
            .field("layers", &self.layers)
            .field("weight", &self.weight)
            .finish()
    }
}
