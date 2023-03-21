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

pub fn calc_loss_auto<T>(iy: usize, emb: &[T], params: &[T]) -> T 
    where T: 'static+Float+Debug+From<f64>+ClosedAdd+ClosedMul+DivAssign, 
{
    let emb = DMatrix::from_column_slice(1, 30, &emb); 
    let w1 = DMatrix::from_column_slice(1, 30, &params[0..30]); 
    let w2 = DMatrix::from_column_slice(1, 27, &params[30..]);

    let dot_prod = emb.dot(&w1); 
    let h = dot_prod.tanh();
    let logits = DMatrix::from_iterator(1, 27, w2.iter().map(|&x| x*h));
    let probs = softmax(&logits);  
    let loss = -probs[(0, iy)].ln();
    println!("loss {loss:?}");
    loss.into()
}

pub fn calc_grad_auto(iy: usize, emb: &[f64], params: &[f64]) -> Vec<f64> {
    let emb_dual: Vec<F<f64, f64>> = emb.iter().map(|&x| F1::var(x)).collect();
    grad(|x| calc_loss_auto(iy, emb_dual.as_slice(), x), params)
}