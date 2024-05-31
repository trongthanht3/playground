use core::fmt;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
use rand_isaac::isaac64::Isaac64Rng;

pub trait Layer {
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, x: Array2<f32>) -> Array2<f32>;
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl fmt::Debug for dyn Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(f)
    }
}

#[derive(Debug, Default)]
pub struct LayerProps {
    pub input: Array2<f32>,
    pub output: Array2<f32>,
}

#[derive(Debug, Default)]
pub struct Dense {
    pub layer_properties: LayerProps,

    pub n_inputs: usize,
    pub n_neurons: usize,

    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
}

impl Dense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Dense {
        let n_inputs = n_inputs;
        let n_neurons = n_neurons;

        // Initialize random weights and bias
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        return Dense {
            layer_properties: LayerProps {
                input: Array2::zeros((1, n_inputs)),
                output: Array2::zeros((1, n_neurons)),
            },
            n_inputs,
            n_neurons,
            weights: Array2::random_using((n_inputs, n_neurons), Uniform::new(0.0, 1.0), &mut rng),
            bias: Array2::zeros((1, n_neurons)),
        };
    }

    pub fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dense")
            .field("layer_properties", &self.layer_properties)
            .field("n_inputs", &self.n_inputs)
            .field("n_neurons", &self.n_neurons)
            .field("weights", &self.weights)
            .field("bias", &self.bias)
            .finish()
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        self.layer_properties.input = x.clone();
        self.layer_properties.output = &x.dot(&self.weights) + &self.bias;
        return self.layer_properties.output.clone();
    }

    fn backward(&mut self, d_value: Array2<f32>) -> Array2<f32> {
        let mut d_weights = self.layer_properties.input.clone().t().dot(&d_value);
        let mut d_bias = d_value.clone().sum_axis(Axis(0));

        let d_inputs = d_value.dot(&self.weights.t());
        return d_inputs;
    }

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dense {{ layer_properties: {:?}, n_inputs: {}, n_neurons: {}, weights: {:?}, bias: {:?} }}",
            self.layer_properties, self.n_inputs, self.n_neurons, self.weights, self.bias
        )
    }
}
