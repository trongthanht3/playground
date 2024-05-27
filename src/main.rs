use ndarray::{prelude::*, Data};
use polars::chunked_array::iterator::par;
use polars::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::array;
use std::collections::HashMap;
use std::path::PathBuf;

fn read_csv(file_path: &str) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(file_path.into()))
        .unwrap()
        .finish()
        .unwrap();

    let train_set = data.drop_many(&["y"]);
    let train_label = data.select(["y"])?;
    return Ok((train_set, train_label));
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    return df
        .to_ndarray::<Float32Type>(IndexOrder::default())
        .unwrap()
        .reversed_axes();
}

struct NeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

impl NeuralNetwork {
    // Init
    pub fn init_params(&self) -> HashMap<String, Array2<f32>> {
        let w_rand = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let num_of_layers = self.layers.len();

        let mut params: HashMap<>
    }
}

fn main() {
    println!("Hello, world!");

    // Load data with read_csv
    let (train_set, train_label) = read_csv("training_set.csv").unwrap();

    let train_nda = array_from_dataframe(&train_set);

    println!("{:?}", train_nda);
}
