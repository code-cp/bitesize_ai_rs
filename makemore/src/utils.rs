use nalgebra::{DMatrix, Scalar};
use num::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign};

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