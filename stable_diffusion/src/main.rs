use burn_autodiff::ADBackendDecorator; 
use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 

use stable_diffusion::train; 

fn main() {
    let device = NdArrayDevice::Cpu; 
    train::run::<ADBackendDecorator<NdArrayBackend<f32>>>(device); 
}
