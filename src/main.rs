extern crate nalgebra as na;
use na::{DMatrix};

extern crate mnist;
use mnist::{Mnist, MnistBuilder};
use rand::distributions::{Normal, Distribution};
use rand::prelude::*;

pub fn create_linear_layer(size_of_inputs: usize, number_of_weights: usize,) -> DMatrix<f32> {
    return DMatrix::from_vec(size_of_inputs, number_of_weights, vec![0.0f32; number_of_weights * size_of_inputs]);
}

pub fn linear_layer(size_of_inputs: usize, number_of_weights: usize) -> DMatrix<f32> {
    let mut zeroed_layer = create_linear_layer(size_of_inputs, number_of_weights);
    let normal = Normal::new(0.0, 1.0 / (1.0 / (number_of_weights * size_of_inputs) as f64).sqrt());
    zeroed_layer.map(|_| normal.sample(&mut rand::thread_rng()) as f32)
}

/// Runs the network forward, return the output of each layer in the network, and the input to network itself
fn forward(input: &DMatrix<f32>, network: &Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
    let mut inputs = vec![input.clone()];
    //Foreach layer in the network
    for n in network {
        //Multiply the last input to be placed in the inputs array by the weights of the network
        let result = &inputs[inputs.len() - 1] * n;
        //Push it through its nonlineraity
        let rectified_ouput = result.map(|x|{
            if x > 0.0f32 {
                return x;
            }
            return 0f32;
        });
        //place into the inputs
        inputs.push(rectified_ouput);
    }
    //Return the ouputs of each layer
    return inputs;
}

fn backword(expected: &DMatrix<f32>, actual: &DMatrix<f32>, network: &Vec<DMatrix<f32>>, outputs: &Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
    //Mean Squared difference.... not really
    let intial_difference = (expected - actual).map(|x| x * x);

    let mut errors = vec![intial_difference];
    let mut deltas = vec![];

    
    //Foreach output from the network, calcualte the derivate
    let mut count = 0;
    for n in network.iter().rev() {
        let derivative_of_output = outputs[(outputs.len() - 1) - count].map(|x| {
            if x > 0.0f32 {
                return 1.0f32;
            }
            else if x == 0.0f32 {
                return 0.5f32;
            }
            return 0.0f32;
        });
        
        //multiply each error by its dervaitve
        let delta = derivative_of_output.zip_map(&errors[errors.len() -1], |a, b| a * b);
        deltas.push(delta);

        //Calculate the error for this layer, to be used by the next layer
        let error = n * deltas[deltas.len() - 1].transpose();
        errors.push(error.transpose());
        count += 1;
    }

    return deltas;
}

fn update_weights(input: &DMatrix<f32>, layer: &mut DMatrix<f32>, delta: &DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>{
    let adjusted_deltas = (delta.transpose().map(|x| x * learning_rate) * input).transpose();    
    return layer.zip_map(&adjusted_deltas, |a, b| a + b);
}

fn train(test_inputs: Vec<DMatrix<f32>>, network: &mut Vec<DMatrix<f32>>, expcted_outputs: Vec<DMatrix<f32>>, training_loops: usize)  {
    let mut rng = rand::thread_rng();
    for i in 0..training_loops {
        let example_index = rng.gen_range(0, test_inputs.len());
        println!("Starting loops {} of training", i);
        let mut inputs = forward(&(test_inputs[example_index]), &network);
        let mut per_layer_deltas = backword(&expcted_outputs[example_index], &inputs[inputs.len() - 1], &network, &inputs);
        let per_layer_deltas : Vec<_> = per_layer_deltas.iter_mut().map(|x|x).rev().collect();
        let _ = inputs.pop();
        for i in 0..inputs.len() {
            update_weights(&inputs[i], &mut network[i], &per_layer_deltas[i], 0.0f32);
        }
    }
}

fn validate_network() {
    
}

fn load_data() -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>, Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
    let (trn_size, val_size, rows, cols) = (50_000, 10000, 28, 28);
    let Mnist { trn_img, trn_lbl, val_img, val_lbl, ..} = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(val_size)
        .test_set_length(10_000)
        .finalize();



    //TRAINING SET
    //Conver the images into single sets of data, and map them from 0-255 space into 0.0 - 1.0 space
    //And flatten out the images themselves, as this is a for a dense network that handles all of the images as a flat buffer
    let mut training_data = vec![];
    let sub_section_size = rows * cols;
    for i in 0..trn_size {
        let offset = (i * sub_section_size) as usize;
        let subsection = &trn_img[offset..(offset + sub_section_size as usize)];
        //We need the data in 0.0 -> 1.0f32 hence the map at the end
        let single_training_input = DMatrix::from_vec(1, rows as usize * cols as usize, subsection.to_vec()).map(|x| (x as f32) / 255.0f32);
        training_data.push(single_training_input);
    }

    //One hot encode all of the labels
    let mut training_labels = vec![];
    for train_label_index in 0..trn_lbl.len() {
        let mut label_vec = vec![0.0f32; 10];
        label_vec[trn_lbl[train_label_index] as usize] = 1.0f32;
        training_labels.push(DMatrix::from_vec(1, 10, label_vec));
    }

    //VALIDATION SET
    //Conver the images into single sets of data, and map them from 0-255 space into 0.0 - 1.0 space
    //And flatten out the images themselves, as this is a for a dense network that handles all of the images as a flat buffer
    let mut validation_data = vec![];
    let sub_section_size = rows * cols;
    for i in 0..val_size {
        let offset = (i * sub_section_size) as usize;
        let subsection = &val_img[offset..(offset + sub_section_size as usize)];
        //We need the data in 0.0 -> 1.0f32 hence the map at the end
        let single_training_input = DMatrix::from_vec(1, rows as usize * cols as usize, subsection.to_vec()).map(|x| (x as f32) / 255.0f32);
        validation_data.push(single_training_input);
    }

    //One hot encode all of the labels of the validation test set
    let mut validation_labels = vec![];
    for train_label_index in 0..val_lbl.len() {
        let mut label_vec = vec![0.0f32; 10];
        label_vec[val_lbl[train_label_index] as usize] = 1.0f32;
        validation_labels.push(DMatrix::from_vec(1, 10, label_vec));
    }
    
    return (training_data, training_labels, validation_data, validation_labels);
}

fn main() {
    //Load the data
    let (training_data, training_labels, validation_data, validation_labels) = load_data();
    let mut network = vec![linear_layer(28 * 28, 128), linear_layer(128, 10)];
    train(training_data, &mut network, training_labels, 100);    
}
