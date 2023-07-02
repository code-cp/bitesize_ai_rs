use burn::{
    config::Config,
    tensor::{backend::Backend, Shape, Data, ElementConversion, Int, Tensor, Distribution},
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset}, 
};
use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 

use ndarray::{Array, Array1, Array2, Array3, s, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use image::{ImageBuffer};

use crate::data::MNISTBatcher;
use crate::model::UNet; 

// type B = NdArrayBackend<f32>;
// type B = burn_autodiff::ADBackendDecorator<NdArrayBackend<f32>>;
// type D = NdArrayDevice;

use burn_tch::{TchBackend, TchDevice}; 
type B = burn_autodiff::ADBackendDecorator<TchBackend<f32>>;
type D = TchDevice;

pub struct DDPM {
    pub betas: Array1<f32>, 
    pub alphas: Array1<f32>,
    pub alpha_bars: Array1<f32>, 
    pub coef1: Array1<f32>, 
    pub coef2: Array1<f32>, 
    pub n_steps: usize,
}

impl DDPM {
    pub fn new(n_steps: usize) -> Self {
        let min_beta: f32 = 0.0001; 
        let max_beta: f32 = 0.02; 

        let betas = Array1::linspace(min_beta, max_beta, n_steps);
        let alphas = 1.0 - &betas;
        let mut alpha_bars: Array1<f32> = Array::zeros(alphas.dim());
        let mut product = 1.0; 

        for (i, alpha) in alphas.iter().enumerate() {
            product *= alpha; 
            alpha_bars[i] = product; 
        }        

        let mut alpha_prev: Array1<f32> = Array1::zeros(alpha_bars.dim());
        alpha_prev.slice_mut(s![1..]).assign(&alpha_bars.slice(s![0..n_steps-1]));
        alpha_prev[0] = 1.0; 

        let sqrt_alphas = alphas.mapv(|x| x.sqrt()); 
        let coef1 = sqrt_alphas * (1.0 - alpha_prev.clone()) / (1.0 - alpha_bars.clone());
        let sqrt_alpha_prev = alpha_prev.mapv(|x| x.sqrt()); 
        let coef2 = sqrt_alpha_prev * betas.clone() / (1.0 - alpha_bars.clone()); 

        DDPM {
            betas, 
            alphas, 
            alpha_bars, 
            coef1, 
            coef2, 
            n_steps, 
        }
    }

    pub fn sample_forward(&self, x: Array2<f32>, t: usize, eps: Option<Array2<f32>>) -> Array2<f32> {
        let alpha_bar = self.alpha_bars[t]; 

        let eps = match eps {
            Some(value) => value, 
            None => Array::random(x.dim(), StandardNormal), 
        };

        let res = eps * (1.0-alpha_bar).sqrt() + alpha_bar.sqrt() * x; 
        res         
    }

    pub fn sample_backward(&self, net: UNet) -> burn::tensor::Tensor<B, 4> {
        let batch_size = 32; 
        let height = 28; 
        let width = 28; 

        let mut x = Tensor::<B, 4>::random([batch_size, 1, height, width], Distribution::Standard);
        for t in self.n_steps..=0 {
            x = self.sample_backward_step(&mut x, t, &net); 
        }

        return x; 
    }

    pub fn sample_backward_step(&self, x_t: &mut Tensor<B, 4>, t: usize, net: &UNet) -> Tensor<B, 4> {
        let eps = net.forward(x_t.to_owned(), t); 
    
        let mut noise: Tensor<B, 4>; 
        if t == 0 {
            noise = Tensor::zeros(x_t.shape()); 
        } else {
            let var = self.betas[t];  
            noise = Tensor::random(x_t.shape(), Distribution::Normal(0.0, 1.0)); 
            noise = noise.mul_scalar(var.sqrt());
        }

        // mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) 
        //     / torch.sqrt(self.alphas[t])
        let coef = 1.0 - self.alphas[t]; 
        let coef = coef / (1.0 - self.alpha_bars[t]).sqrt(); 
        let x = x_t.clone().sub(eps.mul_scalar(coef)); 
        let mean = x.div_scalar(self.alphas[t].sqrt());

        mean + noise 
    }
}

pub fn visualize_forward() {
    let n_steps = 100; 

    // let device: NdArrayDevice = D::Cpu; 
    let device = TchDevice::Mps; 
    let batcher_train: MNISTBatcher<B> = MNISTBatcher::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(3)
        .shuffle(42)
        .num_workers(1)
        .build(MNISTDataset::train());

    let batch = dataloader_train.iter().next().unwrap(); 
    let tensor: burn::tensor::Data<f32, 3> = batch.images.into_data();
    let shape = tensor.shape; 
    let arr = Array3::from_shape_vec((shape.dims[0], shape.dims[1], shape.dims[2]), tensor.value).unwrap();

    let ddpm = DDPM::new(n_steps); 
    let percents = Array1::linspace(0.0, 0.99, 10);
    
    // iterate along the first dimension 
    for (i, x) in arr.axis_iter(Axis(0)).enumerate() {
        for (j, &p) in percents.iter().enumerate() {
            let t = (n_steps as f32) * p; 
            let x_t = ddpm.sample_forward(x.to_owned(), t as usize, None); 
    
            let (height, width) = x_t.dim();
            let x_t = x_t.mapv(|pixel| ((pixel+1.0) / 2.0 * 255.0) as u8);
            let image_buffer = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
                let pixel = x_t[[y as usize, x as usize]];
                image::Luma([pixel])
            });
    
            image_buffer.save(format!("./images/forward_{i:02}_{j:02}.png")).unwrap();
        } 
    }
 

}

mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        visualize_forward(); 
    }
}