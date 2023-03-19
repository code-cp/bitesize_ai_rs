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
use makemore::utils::softmax; 

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
    
    let n1 = (0.8*(words.len() as f32)) as usize; 
    let n2 = (0.9*(words.len() as f32)) as usize; 

    let (x_tr, y_tr) = build_dataset(words[..n1].to_vec(), block_size, &stoi)?; 
    let (x_dev, y_dev) = build_dataset(words[n1..n2].to_vec(), block_size, &stoi)?;
    let (x_te, y_te) = build_dataset(words[n2..].to_vec(), block_size, &stoi)?;

    // the dimensionality of the character embedding vectors
    let n_embd = 10;
    // the number of neurons in the hidden layer of the MLP
    let n_hidden = 200;

    let c: DMatrix<f32> = DMatrix::from_fn(vocab_size, n_embd, |i_, j_| StandardNormal.sample(&mut rng));

    // Layer 1
    let w1: DMatrix<f32> = DMatrix::from_fn(n_embd*block_size, n_hidden, |i_, j_| StandardNormal.sample(&mut rng)) * (5.0/3.0)/((n_embd as f32).powf(0.5));
    // using b1 just for fun, it's useless because of BN
    let b1: DMatrix<f32> = DMatrix::from_fn(1, n_hidden, |i_, j_| StandardNormal.sample(&mut rng))*0.1;

    // Layer 2
    let w2: DMatrix<f32> = DMatrix::from_fn(n_hidden, vocab_size, |i_, j_| StandardNormal.sample(&mut rng)) * 0.1;
    let b2: DMatrix<f32> = DMatrix::from_fn(1, vocab_size,|i_, j_| StandardNormal.sample(&mut rng))*0.1;

    let epoch = 1; 

    for i_ in 0..epoch {
        // forward pass
        let ix = rng.gen_range(0..x_tr.nrows() as usize);
        let emb: DMatrix<f32> = c.select_rows(x_tr.select_rows(&[ix]).as_slice().iter().map(|&x| x as usize).collect::<Vec<usize>>().as_slice());
        // println!("emb size {:?}", emb.shape()); 
        // Reshape the matrix to have -1 in the first dimension and 30 in the second dimension
        let emb = emb.reshape_generic(Dyn(1), Dyn(30));
        // println!("emb size after reshape {:?}", emb.shape());

        let h = (&emb * &w1 + &b1).map(|x: f32| x.tanh());
        // println!("shape of layer1 output {:?}", h.shape()); 

        let logits = &h * &w2 + &b2;
        let probs = softmax(&logits);   
        // println!("shape of layer2 output {:?}", logits.shape());
        // println!("logits {logits:?}");
        // println!("probs {probs:?}");  

        let loss = -probs[(0, y_tr.select_rows(&[ix]).as_slice()[0] as usize)];
        println!("loss is {loss:?}"); 
    }

    Ok(())
}