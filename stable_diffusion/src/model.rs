use crate::data::MNISTBatch;

use burn::{
    module::Module,
    nn::{self, conv::Conv2dPaddingConfig, loss::CrossEntropyLoss, BatchNorm, EmbeddingConfig},
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
        module::conv_transpose2d, 
        ops::ConvTransposeOptions, 
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    embedding: nn::Embedding<B>,
    max_seq_length: usize, 
    batch_size: usize,  
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(batch_size: usize, max_seq_length: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(max_seq_length, d_model).init(); 
        Self {
            embedding,
            max_seq_length, 
            batch_size,  
        }
    }

    pub fn forward(&self, t: usize) -> Tensor<B, 3> {
        // ref https://github.com/burn-rs/burn/blob/f99fe0faddfa9152f0049532b94ce33177dd55f9/examples/text-generation/src/model.rs
        let device = self.devices()[0]; 
        let index_positions = Tensor::arange_device(0..self.max_seq_length, &device)
            .reshape([1, self.max_seq_length])
            .repeat(0, self.batch_size);
        self.embedding.forward(index_positions)
    }
}

#[derive(Module, Debug)]
pub struct UNetBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>, 
    conv2: nn::conv::Conv2d<B>, 
    residual_conv: nn::conv::Conv2d<B>, 
    activation: nn::ReLU,
    layer_norm: nn::LayerNorm<B>, 
}

impl<B: Backend> UNetBlock<B> {
    pub fn new(in_ch: usize, out_ch: usize) -> Self {
        let kernel_size = [3,3];  
        let conv1 = nn::conv::Conv2dConfig::new([in_ch, out_ch], kernel_size)
            .with_padding(Conv2dPaddingConfig::Valid)
            .init(); 
        let conv2 = nn::conv::Conv2dConfig::new([out_ch, out_ch], kernel_size)
            .with_padding(Conv2dPaddingConfig::Valid)
            .init(); 

        let residual_conv = nn::conv::Conv2dConfig::new([in_ch, out_ch], [1,1]).init();
        let layer_norm = nn::LayerNormConfig::new(28).init(); 

        Self {
            conv1, 
            conv2,
            residual_conv, 
            activation: nn::ReLU::new(),  
            layer_norm, 
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.layer_norm.forward(input); 
        let x = self.conv1.forward(x); 
        let x = self.activation.forward(x); 
        let x = self.conv2.forward(x); 
        let x = x.add(self.residual_conv.forward(input));
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    blocks: Vec<UNetBlock<B>>,
    downsampling: Vec<nn::pool::MaxPool2d>, 
    embedding: PositionalEmbedding<B>, 
}

impl<B: Backend> Encoder<B> {
    pub fn new(channels: Vec<usize>, embedding: PositionalEmbedding<B>) -> Self {
        let blocks: Vec<UNetBlock<B>> = (0..channels.len()-1)
            .map(|i| UNetBlock::new(channels[i], channels[i+1]))
            .collect();
        let kernel_size = [2; 2]; 
        let downsampling: Vec<nn::pool::MaxPool2d> = (1..channels.len())
            .map(|i| nn::pool::MaxPool2dConfig::new(i, kernel_size).init())
            .collect(); 
        Encoder {
            blocks, 
            downsampling,
            embedding 
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut xs = Vec::new(); 
        let kernel_size = 2;
        for (&block, &pool) in self.blocks.iter().zip(self.downsampling.iter()) {
            let x = block.forward(x);
            xs.push(x); 
            let x = pool.forward(x); 
        }
        xs 
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    channels: Vec<usize>, 
    blocks: Vec<UNetBlock<B>>,
    embedding: PositionalEmbedding<B>, 
}

impl<B: Backend> Decoder<B> {
    pub fn new(channels: Vec<usize>, embedding: PositionalEmbedding<B>) -> Self {
        let blocks = (0..channels.len()-1)
            .map(
                |i| UNetBlock::new(channels[i], channels[i+1]) 
            )
            .collect();
        Decoder {
            channels, 
            blocks, 
            embedding, 
        }
    }

    pub fn crop(&self, x: Tensor<B, 4>, encoder_feature: Tensor<B, 4>) -> Tensor<B, 4> {
        let new_height = x.shape().dims[2]; 
        let new_width = x.shape().dims[3];

        let height = encoder_feature.shape().dims[2]; 
        let width = encoder_feature.shape().dims[3];

        // Calculate the starting indices for cropping
        let start_h = (height - new_height) / 2;
        let start_w = (width - new_width) / 2;

        // Perform the crop
        encoder_feature.index([start_h..start_h + height, start_w..start_w + width]).clone()
    }

    pub fn forward(&self, x: Tensor<B, 4>, encoder_features: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        for i in 0..self.channels.len()-1 {
            // NOTE, there is no convtranspose2d layer in burn 
            // ref https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html?highlight=convtranspose2d#torch.nn.ConvTranspose2d
            // padding controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding
            // ref https://pytorch.org/docs/stable/generated/torch.nn.functional.conv_transpose2d.html?highlight=conv_transpose2d#torch.nn.functional.conv_transpose2d
            // for weight size 
            let weight = self.blocks[i].conv1.into_record().weight.detach(); 
            let x = conv_transpose2d(x, weight, None, 
                ConvTransposeOptions::new(
                    [2, 2],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    1,
                ),
            ); 
            let feature = self.crop(x, encoder_features[i]); 
            let x = Tensor::cat(vec![x, feature], 1); 
            let x = self.blocks[i].forward(x); 
        }
        x
    } 
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    encoder: Encoder<B>, 
    decoder: Decoder<B>, 
}

impl<B: Backend> UNet<B> {
    pub fn new(n_steps: usize, batch_size: usize, en_chs: Vec<usize>, de_chs: Vec<usize>) -> Self {
        let pe_dim = 128; 
        let embedding = PositionalEmbedding::new(batch_size, n_steps, pe_dim); 
        
        let encoder = Encoder::<B>::new(en_chs, embedding);
        let decoder = Decoder::<B>::new(de_chs, embedding); 

        Self {
            encoder, 
            decoder,  
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut encoder_features = self.encoder.forward(x);
        encoder_features.reverse(); 
        let x = self.decoder.forward(encoder_features[0], encoder_features[1..].to_owned()); 
        x 
    }
}