use crate::lib::layer::Layer;
use core::fmt;
use ndarray::prelude::*;
use std::collections::HashMap;

trait NeuralNetworkFunctions {
    fn loss_function(&self, y_pred: Array2<f32>, y_true: Array2<f32>) -> f32;
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
    pub weight: HashMap<String, Array2<f32>>,
    // pub loss_function: fn(Array2<f32>, Array2<f32>),
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        NeuralNetwork {
            layers: Vec::<Box<dyn Layer>>::new(),
            weight: HashMap::new(),
            // loss_function: NeuralNetwork::loss_function,
        }
    }
}

impl NeuralNetwork {
    // Init
    // pub fn init_params(&self) -> HashMap<String, Array2<f32>> {
    // }

    pub fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNetwork")
            .field("layers", &self.layers)
            .field("weight", &self.weight)
            .finish()
    }

    // Add layers
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn compile(&mut self) {}
}
