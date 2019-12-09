import torchreid
import torch

import argparse

parser = argparse.ArgumentParser(description='Train baseline model for re-identification')
parser.add_argument('--model_type', type=str, help='Model architecture to train', default='squeezenet1_0', choices=["mobilenetv2_x1_4", "squeezenet1_0", "resnet50"])
parser.add_argument('--dataset', type=str, help='Dataset to train on', default='market1501', choices=['dukemtmcreid', 'market1501', 'viper'])
parser.add_argument('--num_epochs', type=int, help='Number of epochs to train on', default=30)

args = parser.parse_args()

model_type = args.model_type
dataset = args.dataset
num_epochs = args.num_epochs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources=dataset,
    targets=dataset,
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

baseline_model = torchreid.models.build_model(
    name=model_type,
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

if torch.cuda.is_available():
    baseline_model = baseline_model.cuda()

print(f"Number of parameters in {model_type} - {count_parameters(baseline_model)}")

optimizer = torchreid.optim.build_optimizer(
    baseline_model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    baseline_model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir=f'log/{model_type}',
    max_epoch=num_epochs,
    eval_freq=10,
    print_freq=10,
    test_only=False
)

torch.save(baseline_model.state_dict(), f"{model_type}.pth")