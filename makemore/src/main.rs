use anyhow;
use std::fs;
use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rand::distributions::{Distribution, Uniform};
use rand::{prelude::*, distributions::WeightedIndex};

use nalgebra::{DMatrix, SMatrix, SVector, Const, U1, Dyn};

use makemore::data::build_dataset;
use makemore::utils::{softmax, calc_grad_auto, calc_loss_auto};

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
    // let n2 = (0.9*(words.len() as f64)) as usize; 

    let (x_tr, y_tr) = build_dataset(words[..n1].to_vec(), block_size, &stoi)?; 
    // let (x_dev, y_dev) = build_dataset(words[n1..n2].to_vec(), block_size, &stoi)?;
    // let (x_te, y_te) = build_dataset(words[n2..].to_vec(), block_size, &stoi)?;

    let batch_size = 32; 
    // the dimensionality of the character embedding vectors
    let n_embd = 10;
    // the number of neurons in the hidden layer of the MLP
    let n_hidden = 200;

    // params contain c, w1, w2 
    let mut params: Vec<f64> = (0..(vocab_size*n_embd+n_embd*block_size*n_hidden+vocab_size*n_hidden)).map(|_| StandardNormal.sample(&mut rng)).collect();

    let learning_rate: f64 = 0.1; 
    let epoch = 2000; 

    for i in 0..epoch {
        println!("epoch {i:?}");
        // forward pass
        let mut ix = Vec::new(); 
        for _ in 0..batch_size {
            let mut rng = StdRng::from_entropy();
            let x = rng.gen_range(0..x_tr.nrows() as usize);
            ix.push(x); 
        }
        let ixs = x_tr.select_rows(ix.as_slice()).as_slice().iter().map(|&x| x as usize).collect::<Vec<usize>>();
        let iy = y_tr.select_rows(ix.as_slice()).as_slice().iter().map(|&x| x as usize).collect::<Vec<usize>>(); 
        let loss = calc_loss_auto(vocab_size, n_embd, n_hidden, &iy, &ixs, &params); 
        println!("loss {loss:?}"); 

        // back prop
        // Automatic differentiation using dual numbers 
        let grad = calc_grad_auto(vocab_size, n_embd, n_hidden, &iy, &ixs, params.as_slice());
        // println!("grad {grad:?}"); 
        params = params.iter().zip(grad.iter()).map(|(&p, &g)| p - g*learning_rate).collect::<Vec<f64>>();
    }

    // sample 
    let c = DMatrix::from_column_slice(vocab_size, n_embd, &params[0..vocab_size*n_embd]); 
    let w1 = DMatrix::from_column_slice(30, n_hidden, &params[vocab_size*n_embd..vocab_size*n_embd+30*n_hidden]); 
    let w2 = DMatrix::from_column_slice(n_hidden, 27, &params[vocab_size*n_embd+30*n_hidden..]);

    // the indices for the characters, used for sampling 
    let values: Vec<usize> = (0..vocab_size).collect(); 

    for _ in 0..5 {
        let mut name = Vec::new(); 
        let mut context = vec![0; block_size];
        loop {
            let emb: DMatrix<f64> = c.select_rows(context.as_slice());
            let emb = emb.reshape_generic(Dyn(1), Dyn(30));
            // println!("emb shape {:?}", emb.shape()); 
            let layer1 = emb * w1.clone(); 
            let h = layer1.map(|x| x.tanh());
            let logits = h * w2.clone(); 
            let probs = softmax(&logits);  

            let dist = WeightedIndex::new(&probs).unwrap();
            let mut rng = StdRng::from_entropy();
            let ix = values[dist.sample(&mut rng)];
    
            context = context[1..].to_vec();
            context.push(ix); 
            name.push(ix); 

            if ix == 0 {
                break; 
            }
        }
        let word = name.iter().map(|i| itos.get(i).unwrap()).collect::<String>();
        println!("name: {word:}");  
    }    

    Ok(())
}

// output after 100 epochs 
// name: kim.
// name: lie.
// name: cianaur.
// name: lee.
// name: lie.

// output after 1000 epochs 
// name: rioh.
// name: muxmiamo.
// name: mire.
// name: mrrms.
// name: gmrz.

// output after 2000 epochs 
// name: taf.
// name: ban.
// name: srhi.
// name: janri.
// name: ury.