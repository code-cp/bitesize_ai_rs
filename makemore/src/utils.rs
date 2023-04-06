use nalgebra::{DMatrix, SMatrix, Scalar, DVector, SVector, Dyn, ClosedMul, ClosedAdd, U1, Const, Matrix, VecStorage};
use num::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign};

use autodiff::*;

pub fn softmax<T>(matrix: &DMatrix<T>) -> DMatrix<T>
where
    T: Float + Debug + AddAssign + DivAssign + Scalar,
{
    let mut exp_matrix = matrix.map(|x| x.exp());
    for i in 0..exp_matrix.nrows() {
        let row_sum = exp_matrix.row(i).sum();
        exp_matrix.set_row(i, &(exp_matrix.row(i) / row_sum));
    }
    exp_matrix
}

pub fn calc_loss_auto<T>(vocab_size: usize, n_embd: usize, n_hidden: usize, iy: &[usize], ixs: &[usize], params: &[T]) -> T 
    where T: 'static+Default+Float+Debug+From<f64>+ClosedAdd+ClosedMul+DivAssign, 
{
    let c = DMatrix::from_column_slice(vocab_size, n_embd, &params[0..vocab_size*n_embd]); 
    let w1 = DMatrix::from_column_slice(30, n_hidden, &params[vocab_size*n_embd..vocab_size*n_embd+30*n_hidden]); 
    let w2 = DMatrix::from_column_slice(n_hidden, 27, &params[vocab_size*n_embd+30*n_hidden..]);

    let emb: DMatrix<T> = c.select_rows(ixs);
    // println!("emb shape {:?}", emb.shape()); 
    let row_num = emb.nrows() * emb.ncols() / 30; 
    let emb = emb.reshape_generic(Dyn(row_num), Dyn(30));

    let layer1 = emb * w1; 
    let h = layer1.map(|x| x.tanh());
    let logits = h * w2; 
    let probs = softmax(&logits);  
    let mut loss = T::zero(); 
    {
        for (i, y) in iy.iter().enumerate() {
            loss += -probs[(i, *y)].ln(); 
        }
    }
    // println!("loss {loss:?}");
    (loss/<T as From<f64>>::from(probs.nrows() as f64)).into()
}

pub fn calc_grad_auto(vocab_size: usize, n_embd: usize, n_hidden: usize, iy: &[usize], ixs: &[usize], params: &[f64]) -> Vec<f64> {
    grad(|x| calc_loss_auto(vocab_size, n_embd, n_hidden, iy, ixs, x), params)
}