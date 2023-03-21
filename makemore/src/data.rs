use anyhow;
use std::collections::HashMap;
use nalgebra::*; 

pub fn build_dataset(words: Vec<&str>, block_size: usize, stoi: &HashMap<char, usize>) -> anyhow::Result<(DMatrix<f64>, DVector<f64>)> {
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    words.iter().for_each(|w| {
        let mut context = vec![0.0; block_size];
        w.chars().chain(".".chars()).for_each(|ch| {
            let ix = stoi[&ch];
            x.push(context.clone());
            y.push(ix as f64);
            context = context[1..].to_vec();
            context.push(ix as f64);
        });
    });
    

    let x_mat = DMatrix::from_row_slice(x.len(), block_size, x.concat().as_slice());
    let y_vec = DVector::from_vec(y);

    println!("{:?}, {:?}", x_mat.shape(), y_vec.shape());
    Ok((x_mat, y_vec))
}