use nalgebra::{DMatrix, Dyn};

pub fn softmax(matrix: &DMatrix<f32>) -> DMatrix<f32> {
    let max_val = matrix.fold(0.0, |max, x| f32::max(max, x)); 
    let mut exp_matrix = matrix.map(|x| (x - max_val).exp());
    for i in 0..exp_matrix.nrows() {
        let row_sum = exp_matrix.row(i).sum();
        exp_matrix.set_row(i, &(exp_matrix.row(i) / row_sum));
    }
    exp_matrix
}
