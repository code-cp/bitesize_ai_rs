use nalgebra::{DMatrix, SMatrix, Scalar, DVector, SVector, Dyn, ClosedMul, ClosedAdd};
use num::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign};

use autodiff::*;

pub fn softmax<T>(matrix: &DMatrix<T>) -> DMatrix<T>
where
    T: Float + Debug + AddAssign + DivAssign + Scalar,
{
    let max_val = matrix.fold(T::zero(), |max, x| T::max(max, x));
    let mut exp_matrix = matrix.map(|x| (x - max_val).exp());
    for i in 0..exp_matrix.nrows() {
        let row_sum = exp_matrix.row(i).sum();
        exp_matrix.set_row(i, &(exp_matrix.row(i) / row_sum));
    }
    exp_matrix
}

pub fn calc_loss_auto<T>(vocab_size: usize, n_embd: usize, iy: usize, ixs: &[usize], params: &[T]) -> T 
    where T: 'static+Default+Float+Debug+From<f64>+ClosedAdd+ClosedMul+DivAssign, 
{
    let c = DMatrix::from_column_slice(vocab_size, n_embd, &params[0..vocab_size*n_embd]); 
    let w1 = DMatrix::from_column_slice(1, 30, &params[vocab_size*n_embd..vocab_size*n_embd+30]); 
    let w2 = DMatrix::from_column_slice(1, 27, &params[vocab_size*n_embd+30..]);

    let emb: DMatrix<T> = c.select_rows(ixs);
    let emb: Vec<T> = emb.iter().map(|&x| x).collect();

    let dot_prod = emb.iter().zip(w1.iter()).map(|(&x, &y)| x * y).fold(T::default(), |acc, x| acc + x); 
    let h = dot_prod.tanh();
    let logits = DMatrix::from_iterator(1, 27, w2.iter().map(|&x| x*h));
    let probs = softmax(&logits);  
    let loss = -probs[(0, iy)].ln();
    println!("loss {loss:?}");
    loss.into()
}

pub fn calc_grad_auto(vocab_size: usize, n_embd: usize, iy: usize, ixs: &[usize], params: &[f64]) -> Vec<f64> {
    grad(|x| calc_loss_auto(vocab_size, n_embd, iy, ixs, x), params)
}