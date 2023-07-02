use crate::data::MNISTBatch;

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

use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 
use burn_train::metric::{Adaptor, LossInput};
use burn_tensor::{loss::cross_entropy_with_logits, Data, Shape, Distribution};

use ndarray::{Array, Array1, Array2, Array3, s, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::ddpm::*; 

// type B = NdArrayBackend<f32>;
type B = burn_autodiff::ADBackendDecorator<NdArrayBackend<f32>>;

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    embedding: nn::Embedding<B>,
    max_seq_length: usize, 
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(max_seq_length: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(max_seq_length, d_model).init(); 
        Self {
            embedding,
            max_seq_length, 
        }
    }

    pub fn forward(&self, t: usize) -> Tensor<B, 3> {
        // ref https://github.com/burn-rs/burn/blob/f99fe0faddfa9152f0049532b94ce33177dd55f9/examples/text-generation/src/model.rs
        let index_positions = Tensor::ones(Shape::new([1, 1])).mul_scalar(t as f32);
        // println!("embed weight shape {:?}", self.embedding.clone().into_record().weight.shape()); 
        let embed = self.embedding.forward(index_positions); 
        // println!("embed weight shape {:?}", embed.shape()); 
        embed 
    }
}

#[derive(Module, Debug)]
pub struct UNetBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>, 
    conv2: nn::conv::Conv2d<B>, 
    activation: nn::ReLU,
    // layer_norm: nn::LayerNorm<B>, 
    residual_conv: nn::conv::Conv2d<B>, 
}

impl<B: Backend> UNetBlock<B> {
    pub fn new(in_ch: usize, out_ch: usize) -> Self {
        let kernel_size = [3,3];  
        let conv1 = nn::conv::Conv2dConfig::new([in_ch, out_ch], kernel_size)
            .with_padding(Conv2dPaddingConfig::Explicit(1,1))
            .init(); 
        let conv2 = nn::conv::Conv2dConfig::new([out_ch, out_ch], kernel_size)
            .with_padding(Conv2dPaddingConfig::Explicit(1,1))
            .init(); 

        // let layer_norm = nn::LayerNormConfig::new(d_model).init(); 
        let residual_conv = nn::conv::Conv2dConfig::new([in_ch, out_ch], [1,1]).init();

        Self {
            conv1, 
            conv2,
            activation: nn::ReLU::new(),  
            // layer_norm, 
            residual_conv, 
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // let x = self.layer_norm.forward(input); 
        let x = self.conv1.forward(input.clone()); 
        let x = self.activation.forward(x); 
        let x = self.conv2.forward(x); 
        let x = x.add(self.residual_conv.forward(input));
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PEEncoderBlock<B: Backend> {
    layer1: nn::Linear<B>, 
    activation: nn::ReLU, 
    layer2: nn::Linear<B>, 
}

impl<B: Backend> PEEncoderBlock<B> {
    pub fn new(pe_dim: usize, channel: usize) -> Self {
        let layer1 = nn::LinearConfig::new(pe_dim, channel).init(); 
        let layer2 = nn::LinearConfig::new(channel, channel).init(); 
        PEEncoderBlock { 
            layer1, 
            activation: nn::ReLU::new(),
            layer2,  
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.layer1.forward(x);
        let x = self.activation.forward(x); 
        let x = self.layer2.forward(x);  
        x
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    blocks: Vec<UNetBlock<B>>,
    downsampling: Vec<nn::pool::MaxPool2d>, 
    embedding: PositionalEmbedding<B>, 
    pe_blocks: Vec<PEEncoderBlock<B>>, 
}

impl<B: Backend> Encoder<B> {
    pub fn new(channels: Vec<usize>, embedding: PositionalEmbedding<B>) -> Self {
        let blocks: Vec<UNetBlock<B>> = (0..channels.len()-1)
            .map(|i| UNetBlock::new(channels[i], channels[i+1]))
            .collect();
        // let kernel_size = [2; 2]; 
        // to make input and output size stay at 28x28 
        let kernel_size = [1; 2]; 
        let downsampling: Vec<nn::pool::MaxPool2d> = (1..channels.len())
            .map(|i| nn::pool::MaxPool2dConfig::new(i, kernel_size).init())
            .collect();
        
        let pe_dim = 128;
        let pe_blocks = (0..channels.len()-1)
            .map(|i| PEEncoderBlock::new(pe_dim, channels[i]))
            .collect(); 

        Encoder {
            blocks, 
            downsampling,
            embedding, 
            pe_blocks,  
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>, t: usize) -> Vec<Tensor<B, 4>> {
        let mut xs = Vec::new(); 
        let mut x = input.clone(); 
        for ((&ref block, &ref pool), &ref pe_layer) in self.blocks.iter().zip(self.downsampling.iter()).zip(self.pe_blocks.iter()) {
            let pe = pe_layer.forward(self.embedding.forward(t));
            let pe = pe.unsqueeze::<4>(); 
            let pe = pe.swap_dims(1, 3);
            // println!("Encoder: x shape {:?} pe shape {:?}", x.shape(), pe.shape()); 
            x = x.clone() + pe; 
            x = block.forward(x);
            xs.push(x.clone()); 
            x = pool.forward(x); 
        }
        // println!("Encoder forward finish"); 
        xs 
    }
}

#[derive(Module, Debug)]
pub struct PEDecoderBlcok<B: Backend> {
    layer1: nn::Linear<B>, 
}

impl<B: Backend> PEDecoderBlcok<B> {
    pub fn new(pe_dim: usize, channel: usize) -> Self {
        let layer1 = nn::LinearConfig::new(pe_dim, channel).init();
        PEDecoderBlcok { layer1 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.layer1.forward(x); 
        x 
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    channels: Vec<usize>, 
    blocks: Vec<UNetBlock<B>>,
    upconvs: Vec<nn::conv::Conv2d<B>>, 
    embedding: PositionalEmbedding<B>, 
    pe_blocks: Vec<PEDecoderBlcok<B>>, 
    conv_out: nn::conv::Conv2d<B>, 
}

impl<B: Backend> Decoder<B> {
    pub fn new(channels: Vec<usize>, embedding: PositionalEmbedding<B>) -> Self {
        let blocks = (0..channels.len()-1)
            .map(
                |i| UNetBlock::new(channels[i], channels[i+1]) 
            )
            .collect();

        // NOTE, there is no convtranspose2d layer in burn, it's backward pass is yet to be implemented 
        let kernel_size = [3; 2]; 
        let upconvs = (0..channels.len()-1).map(
            |i| nn::conv::Conv2dConfig::new([channels[i], channels[i+1]], kernel_size)
            .with_padding(Conv2dPaddingConfig::Explicit(1,1))
            .init()
        ).collect(); 

        let pe_dim = 128;
        let pe_blocks = (1..channels.len())
            .map(|i| PEDecoderBlcok::new(pe_dim, channels[i]))
            .collect(); 

        let conv_out = nn::conv::Conv2dConfig::new([channels[channels.len()-1], 1], kernel_size)
        .with_padding(Conv2dPaddingConfig::Explicit(1,1))
        .init();  

        Decoder {
            channels, 
            blocks, 
            upconvs, 
            embedding,
            pe_blocks,  
            conv_out, 
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>, encoder_features: Vec<Tensor<B, 4>>, t: usize) -> Tensor<B, 4> {
        // println!("Decoder forward begin");  
        let mut x = input.clone(); 
        for i in 0..self.channels.len()-1 {
            x = self.upconvs[i].forward(x); 
            let pe = self.pe_blocks[i].forward(self.embedding.forward(t));
            let pe = pe.unsqueeze::<4>();  
            let pe = pe.swap_dims(1, 3);
            // println!("Decoder: x shape {:?} pe shape {:?}, encoder_feature shape {:?}", x.shape(), pe.shape(), encoder_features[i].shape());  
            x = Tensor::cat(vec![x.clone() + pe, encoder_features[i].clone()], 1); 
            x = self.blocks[i].forward(x); 
        }
        x = self.conv_out.forward(x); 
        x
    } 
}

#[derive(Module, Debug, Clone)]
pub struct UNet {
    encoder: Encoder<B>, 
    decoder: Decoder<B>, 
}

impl std::fmt::Display for UNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UNet {{ encoder: {:?}, decoder: {:?} }}", self.encoder, self.decoder)
    }
}

impl UNet {
    pub fn new(n_steps: usize, en_chs: Vec<usize>, de_chs: Vec<usize>) -> Self {
       let pe_dim = 128;
        let embedding = PositionalEmbedding::new(n_steps, pe_dim); 
        
        let encoder = Encoder::<B>::new(en_chs, embedding.clone());
        let decoder = Decoder::<B>::new(de_chs, embedding.clone()); 

        Self {
            encoder, 
            decoder,  
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t: usize) -> Tensor<B, 4> {
        let mut encoder_features = self.encoder.forward(x, t);
        encoder_features.reverse(); 
        let x = self.decoder.forward(encoder_features[0].clone(), encoder_features[1..].to_owned(), t); 
        x 
    }

    pub fn forward_regression(&self, ddpm: DDPM, item: MNISTBatch<B>) -> RegressionOutput<B> {
        let images = item.images;   
        let mut predictions = Vec::new(); 
        let mut labels = Vec::new(); 

        let batch_size = images.shape().dims[0]; 
        let t_batch = Array::random((batch_size,), Uniform::new(0, ddpm.n_steps));

        for i in 0..batch_size {
            // println!("{:?} th batch", i); 
            let img = images.clone().index([i..i+1, 0..28, 0..28]).squeeze(0);
            // println!("forward regression: img size {:?}", img.shape()); 
            let eps_tensor = Tensor::random(img.shape(), Distribution::Normal(0.0, 1.0)); 
            // println!("forward regression: eps_tensor size {:?}", eps_tensor.shape());
            labels.push(eps_tensor.clone()); 

            let tensor: burn::tensor::Data<f32, 2> = img.into_data();
            let shape = tensor.shape; 
            let x = Array2::from_shape_vec((shape.dims[0], shape.dims[1]), tensor.value).unwrap();

            let tensor: burn::tensor::Data<f32, 2> = eps_tensor.into_data();
            let shape = tensor.shape; 
            let eps = Array2::from_shape_vec((shape.dims[0], shape.dims[1]), tensor.value).unwrap();

            // println!("forward pass");  
            let t = t_batch[i]; 
            let x_ndarray = ddpm.sample_forward(x, t, Some(eps));
            // convert ndarray to tensor 
            let x_vec: Vec<f32> = x_ndarray.iter().map(|x| *x).collect();
            let x_data = Data::from(x_vec.as_slice()); 
            let x_tensor = Tensor::from_data(x_data); 
            let x_t = x_tensor.reshape(Shape::new([28, 28]));
            let x_t = x_t.unsqueeze::<4>(); 
            
            // println!("forward regression: x_t size {:?}", x_t.shape()); 
            let eps_theta: Tensor<B, 4> = self.forward(x_t, t); 
            // println!("forward regression: eps_theta size {:?}", eps_theta.shape()); 
            let eps_theta: Tensor<B, 3> = eps_theta.squeeze(0);
            let eps_theta: Tensor<B, 2> = eps_theta.squeeze(0); 
            predictions.push(eps_theta); 
        }

        let output = Tensor::cat(predictions, 0); 
        let targets = Tensor::cat(labels, 0); 
        let loss = MSELoss::new();
        let loss = loss.forward(output.clone(), targets.clone(), Reduction::Mean);    

        // println!("output shape {:?}", output.shape()); 
        // println!("targets shape {:?}", targets.shape());

        RegressionOutput { loss, output, targets } 
    }
}

impl TrainStep<MNISTBatch<B>, RegressionOutput<B>> for UNet {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let ddpm = DDPM::new(1000);
        let item = self.forward_regression(ddpm, item); 
        TrainOutput::new::<B, UNet>(self, item.loss.backward(), item)
    }
}

impl ValidStep<MNISTBatch<B>, RegressionOutput<B>> for UNet {
    fn step(&self, item: MNISTBatch<B>) -> RegressionOutput<B> {
        let ddpm = DDPM::new(1000);
        self.forward_regression(ddpm, item)
    }
}