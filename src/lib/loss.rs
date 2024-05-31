use crate::lib::layer::Layer;
use core::fmt;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
use rand_isaac::isaac64::Isaac64Rng;
use std::collections::btree_map::Range;
use std::f32;

trait Loss {
    fn forward(&mut self, y_pred: Array2<f32>, y_true: Array2<f32>) -> f32;
    fn backward(&mut self, d_values: Array2<f32>, y_true: Array2<f32>) -> Array2<f32>;
    fn caculate_loss(&self, output: Array2<f32>, y: Array2<f32>) -> f32;
    fn caculate_accumulated_loss(&self) -> f32;
}

// fn cross_entropy_loss(y_pred: Array2<f32>, y_true: Array2<f32>) -> f32 {
//     let n_samples = y_true.shape()[0] as f32;
//     let mut output = -1.0 / n_samples * y_true.mapv(|x| x * y_pred.mapv(|x| x.ln()).sum());
//     return output;
// }


// All this shit is suck, and I don't really feed good now