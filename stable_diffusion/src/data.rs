use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct MNISTBatcher<B: Backend>{
    pub device: B::Device, 
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    // ref https://github.com/burn-rs/burn/blob/f99fe0faddfa9152f0049532b94ce33177dd55f9/examples/mnist/src/data.rs#L12C30-L12C30
    // images are NOT arranged as b, c, w, h 
    pub images: Tensor<B, 3>, 
    pub targets: Tensor<B, 1, Int>, 
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {device}
    }
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items 
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1,28,28]))
            // normalize
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect(); 

        let targets = items 
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect(); 

        let images = Tensor::cat(images, 0).to_device(&self.device); 
        let targets = Tensor::cat(targets, 0).to_device(&self.device); 

        MNISTBatch { images, targets } 
    }
}
