use burn_autodiff::ADBackendDecorator; 
use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 
use burn::{
    module::Module,
    nn::{self, conv::Conv2dPaddingConfig, loss::MSELoss, loss::Reduction, BatchNorm, EmbeddingConfig},
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
        module::conv_transpose2d, 
        ops::ConvTransposeOptions, 
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use burn::record::BinBytesRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

use ndarray::{Array, Array1, Array2, Array3, s, Axis};
use image::{ImageBuffer};

use stable_diffusion::train; 
use stable_diffusion::ddpm::*; 
use stable_diffusion::model::UNet;

// type B = ADBackendDecorator<NdArrayBackend<f32>>; 

use burn_tch::{TchBackend, TchDevice}; 
type B = burn_autodiff::ADBackendDecorator<TchBackend<f32>>;
// type D = TchDevice;

// const MODEL_STATE_FILE_NAME: &str = "./tmp/checkpoint/model-1.mpk.gz";

fn main() {
    // let device = NdArrayDevice::Cpu; 
    let device = TchDevice::Mps; 
    let model_trained = train::run::<B>(device); 

    let n_steps = 1000; 
    // let en_chs = vec![1,64,128,256,512,1024]; 
    // let de_chs = vec![1024, 512, 256, 128, 64]; 
    // let model = UNet::new(n_steps, en_chs, de_chs);
    // let model_trained = model.load_state(MODEL_STATE_FILE_NAME);

    let ddpm = DDPM::new(n_steps);
    let img = ddpm.sample_backward(model_trained); 
    // println!("img shape {:?}", img.shape()); 
    let img: Tensor<B, 4> = img.index([0..1, 0..1, 0..28, 0..28]);
    let img: Tensor<B, 3> = img.squeeze(0); 
    let img: Tensor<B, 2> = img.squeeze(0); 
    let tensor: burn::tensor::Data<f32, 2> = img.into_data();
    let shape: burn_tensor::Shape<2> = tensor.shape; 
    let x_t = Array2::from_shape_vec((shape.dims[0], shape.dims[1]), tensor.value).unwrap();

    let (height, width) = x_t.dim();
    let x_t = x_t.mapv(|pixel| ((pixel+1.0) / 2.0 * 255.0) as u8);
    let image_buffer = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let pixel = x_t[[y as usize, x as usize]];
        image::Luma([pixel])
    });

    image_buffer.save(format!("./images/backward.png")).unwrap();
}
