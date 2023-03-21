use nalgebra::{DMatrix, SMatrix, Scalar, DVector, SVector, Dyn};
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

pub fn calc_loss_auto<T>(ix: usize, y_tr: &DVector<f64>, emb: &DMatrix<f64>, params: &[T]) -> T 
    where T: 'static+Float+Debug+From<f64>, 
{
    let params: Vec<f64> = params.iter().map(|&x| <f64 as num::NumCast>::from(x).unwrap()).collect();
    let params = params.as_slice(); 
    let w1 = DMatrix::from_column_slice(30, 1, &params[0..30]); 
    let w2 = DMatrix::from_column_slice(1, 27, &params[30..]);
    let h = (emb * &w1).map(|x: f64| x.tanh());
    let logits = &h * &w2;
    let probs = softmax(&logits);  
    let loss = -probs[(0, y_tr.select_rows(&[ix]).as_slice()[0] as usize)];
    loss.into()
}

pub fn calc_grad_auto(ix: usize, y_tr: &DVector<f64>, emb: &DMatrix<f64>, params: &[f64]) -> Vec<f64> {
    grad(|x| calc_loss_auto(ix, y_tr, emb, x), params)
}