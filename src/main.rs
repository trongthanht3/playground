mod lib;
use lib::layer::*;
use lib::network::NeuralNetwork;

use ndarray::{prelude::*, Data};
use polars::chunked_array::iterator::par;
use polars::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::array;
use std::collections::HashMap;
use std::default::Default;
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

fn main() {
    println!("Hello, world!");

    // Load data with read_csv
    // let (train_set, train_label) = read_csv("training_set.csv").unwrap();

    // let train_nda = array_from_dataframe(&train_set);

    // println!("{:?}", train_nda);

    // Test network
    let mut simple_network = NeuralNetwork {
        layers: Vec::<Box<dyn Layer>>::new(),
        weight: HashMap::new(),
    };
    println!("{:?}", simple_network);

    let mut layer1 = Dense::new(2, 4);

    let mut rng: ThreadRng = rand::thread_rng();
    let random_set = rng.gen_range(0.0..1.0);
    println!("Random set: {:#?}", &layer1);

    simple_network.add(Box::new(Dense::new(2, 4)));
    println!("{:#?}", simple_network);

    let example_input = array![[1.0], [2.0], [3.0], [4.0]];

    println!("{:#?}", example_input);

    let res_from_layer1 = &layer1.forward(example_input);
    println!("{:#?}", res_from_layer1);
}
