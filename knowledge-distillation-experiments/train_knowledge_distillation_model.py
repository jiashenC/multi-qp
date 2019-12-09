import torchreid
import torch
import argparse

from ImageKnowledgeDistillationEngine import ImageKnowledgeDistillationEngine

parser = argparse.ArgumentParser(description='Train baseline model for re-identification')
parser.add_argument('--model_type', type=str, help='Model architecture to train', default='squeezenet1_0', choices=["mobilenetv2_x1_4", "squeezenet1_0", "resnet50"])
parser.add_argument('--teacher_type', type=str, help='Teacher model to distill from', default='resnet50', choices=["mobilenetv2_x1_4", "squeezenet1_0", "resnet50"])
parser.add_argument('--dataset', type=str, help='Dataset to train on', default='market1501', choices=['dukemtmcreid', 'market1501', 'viper'])
parser.add_argument('--temperature', type=float, help='Temperature for knowledge distillation', default=20)
parser.add_argument('--alpha', type=float, help='Weighting of cross entropy loss / kl divergence loss', default=0.01)
parser.add_argument('--num_epochs', type=int, help='Number of epochs to train on', default=30)

args = parser.parse_args()

model_type = args.model_type
teacher_type = args.teacher_type
temperature = args.temperature
alpha = args.alpha
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

teacher_model = torchreid.models.build_model(
    name=teacher_type,
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

teacher_model.load_state_dict(torch.load(f"{teacher_type}.pth"))

model_distilled = torchreid.models.build_model(
    name=model_type,
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

if torch.cuda.is_available():
    model_distilled = model_distilled.cuda()
    teacher_model = teacher_model.cuda()

print(f"Number of parameters in {teacher_type} - {count_parameters(teacher_model)}")
print(f"Number of parameters in {model_type} - {count_parameters(model_distilled)}")

optimizer = torchreid.optim.build_optimizer(
    model_distilled,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = ImageKnowledgeDistillationEngine(
    datamanager,
    model_distilled,
    optimizer=optimizer,
    scheduler=scheduler,
    teacher_model=teacher_model,
    alpha=alpha,
    temperature=temperature,
    label_smooth=True
)

engine.run(
    save_dir=f"log/{model_type}_distilled",
    max_epoch=num_epochs,
    eval_freq=10,
    print_freq=10,
    test_only=False
)

torch.save(model_distilled.state_dict(), f"{model_type}_distilled.pth")