use anyhow;
use std::fs;
use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rand_distr::Distribution;

use nalgebra::{DMatrix, Dyn};

use makemore::data::build_dataset;
use makemore::utils::{calc_loss_auto, calc_grad_auto};

// implements https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb

fn main() -> anyhow::Result<()> {
    // read in all the words
    let contents = fs::read_to_string("names.txt").expect("Failed to read file");
    let mut words: Vec<&str> = contents.split('\n').collect();
    println!("len of words {:?}", words.len());
    println!("first few words {:?}", &words[0..8]);

    // build the vocabulary of characters and mappings to/from integers
    let mut chars: Vec<char> = words.join("").chars().collect();
    chars.sort();
    chars.dedup();
    let mut stoi: HashMap<char, usize> = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        stoi.insert(*c, i + 1);
    }
    stoi.insert('.', 0);
    let mut itos: HashMap<usize, char> = HashMap::new();
    for (&s, &i) in stoi.iter() {
        itos.insert(i, s);
    }
    let vocab_size = itos.len(); 
    println!("itos {:?}", itos); 
    println!("vocab_size {vocab_size:?}"); 

    // build the dataset
    // context length: how many characters do we take to predict the next one?
    let block_size = 3; 
    let mut rng = StdRng::seed_from_u64(42);
    words.shuffle(&mut rng);
    
    let n1 = (0.8*(words.len() as f64)) as usize; 
    let n2 = (0.9*(words.len() as f64)) as usize; 

    let (x_tr, y_tr) = build_dataset(words[..n1].to_vec(), block_size, &stoi)?; 
    let (x_dev, y_dev) = build_dataset(words[n1..n2].to_vec(), block_size, &stoi)?;
    let (x_te, y_te) = build_dataset(words[n2..].to_vec(), block_size, &stoi)?;

    // the dimensionality of the character embedding vectors
    let n_embd = 10;
    // the number of neurons in the hidden layer of the MLP
    let n_hidden = 1;

    let c: DMatrix<f64> = DMatrix::from_fn(vocab_size, n_embd, |i_, j_| StandardNormal.sample(&mut rng));

    let mut params: Vec<f64> = (0..(n_embd*block_size+vocab_size)).map(|x_| StandardNormal.sample(&mut rng)).collect();

    let learning_rate: f64 = 1e-3; 
    let epoch = 10; 

    for i_ in 0..epoch {
        let mut rng = StdRng::seed_from_u64(42);
        // forward pass
        let ix = rng.gen_range(0..x_tr.nrows() as usize);

        let emb: DMatrix<f64> = c.select_rows(x_tr.select_rows(&[ix]).as_slice().iter().map(|&x| x as usize).collect::<Vec<usize>>().as_slice());
        // println!("emb size {:?}", emb.shape()); 
        // Reshape the matrix to have -1 in the first dimension and 30 in the second dimension
        let emb = emb.reshape_generic(Dyn(1), Dyn(30));
        // println!("emb size after reshape {:?}", emb.shape());
    
        let loss = calc_loss_auto(ix, &y_tr, &emb, params.as_slice());
        println!("loss {loss:?}");

        // back prop 
        let grad = calc_grad_auto(ix, &y_tr, &emb, params.as_slice());
        println!("grad {grad:?}"); 
        params = params.iter().zip(grad.iter()).map(|(&p, &g)| p - g*learning_rate).collect::<Vec<f64>>();
    }

    Ok(())
}