use crate::data::MNISTBatcher;
use crate::model::UNet;
use crate::ddpm::*; 

use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, NoStdTrainingRecorder, Recorder};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::{ADBackend, Backend},
    train::{
        metric::{LossMetric},
        LearnerBuilder,
    },
};

use burn_autodiff::ADBackendDecorator;
use burn_ndarray::{NdArrayBackend, NdArrayDevice}; 

static ARTIFACT_DIR: &str = "./tmp";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 1)]
    pub num_epochs: usize,

    // #[config(default = 256)]
    #[config(default = 1)]
    pub batch_size: usize,

    #[config(default = 32)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: ADBackend<InnerBackend = NdArrayBackend<f32>>>(device: <B as Backend>::Device) -> UNet {
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let batcher_train = MNISTBatcher::<ADBackendDecorator<NdArrayBackend<f32>>>::new(NdArrayDevice::Cpu);
    let batcher_valid = MNISTBatcher::<ADBackendDecorator<NdArrayBackend<f32>>>::new(NdArrayDevice::Cpu);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        // .build(MNISTDataset::train());
        .build(MNISTDataset::test());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::test());

    // Model
    let n_steps = 1000; 
    let batch_size = config.batch_size; 
    let en_chs = vec![1,64,128,256,512,1024]; 
    let de_chs = vec![1024, 512, 256, 128, 64]; 

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(1, CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(UNet::new(n_steps, batch_size, en_chs, de_chs), config.optimizer.init::<B, UNet>(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    NoStdTrainingRecorder::new()
        .record(
            <UNet as Module<B>>::into_record(model_trained.clone()),
            format!("{ARTIFACT_DIR}/model").into(),
        )
        .expect("Failed to save trained model");

    model_trained
}