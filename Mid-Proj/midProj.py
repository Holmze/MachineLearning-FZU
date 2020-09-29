from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import torch.optim as optim
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math
from contextlib import redirect_stdout
from itertools import chain
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine,LinearTransformation
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Subset
from PIL.Image import BICUBIC
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
from apex import amp
from ignite.utils import convert_tensor
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Accuracy, Precision, Recall, Loss, TopKCategoricalAccuracy
#from ignite.contrib.handlers import TensorboardLogger
#from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from datetime import datetime
from ignite.contrib.handlers import CustomPeriodicEvent
import logging
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import ProgressBar
import dill
import pickle 
from ignite.engine import create_supervised_trainer
assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
torch.backends.cudnn.benchmark = True

device = "cuda"

path = "."
image_size = 32

train_transform = Compose([
    RandomCrop(32),
    RandomHorizontalFlip(),
    Pad(4),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #LinearTransformation(transformation_matrix, mean_vector), 
])


test_transform = Compose([ 
    Pad(4),   
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True)
test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=True)

train_eval_indices = [random.randint(0, len(train_dataset) - 1) for i in range(len(test_dataset))]
train_eval_dataset = Subset(train_dataset, train_eval_indices)


batch_size = 100
num_workers = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(17)

class CIFARClassifierELUPaper(nn.Module):

    def __init__(self, num_classes=100):
        super(CIFARClassifierELUPaper, self).__init__()
        
        self.main = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 384, 3, padding=1), # 32
            nn.ELU(),
            #nn.BatchNorm2d(384,eps=1e-4,momentum=0.1),
            nn.MaxPool2d(2, 2), # 16
            # Block 2
            nn.Conv2d(384, 384, 1, padding=0), # 16
            nn.ELU(),
           # nn.BatchNorm2d(384,eps=1e-4,momentum=0.1),
            nn.Conv2d(384, 384, 2, padding=1), # 17
            nn.ELU(),
            #nn.BatchNorm2d(384,eps=1e-4,momentum=0.1),
            nn.Conv2d(384, 640, 2, padding=1), # 18
            nn.ELU(),
            #nn.BatchNorm2d(640,eps=1e-4,momentum=0.1),
            nn.MaxPool2d(2, 2), # 9
            nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(640, 640, 1, padding=0), # 9
            nn.ELU(),
           # nn.BatchNorm2d(640,eps=1e-4,momentum=0.1),
            nn.Conv2d(640, 768, 2, padding=1), # 10
            nn.ELU(),
            #nn.BatchNorm2d(768,eps=1e-4,momentum=0.1),
            nn.Conv2d(768, 768, 2, padding=1), # 11
            nn.ELU(),
            #nn.BatchNorm2d(768,eps=1e-4,momentum=0.1),
            nn.Conv2d(768, 768, 2, padding=1), # 12
            nn.ELU(),
            #nn.BatchNorm2d(768,eps=1e-4,momentum=0.1),
            nn.MaxPool2d(2, 2), # 6
            nn.Dropout2d(0.2),
            # Block 4
            nn.Conv2d(768, 768, 1, padding=0), # 6
            nn.ELU(),
            #nn.BatchNorm2d(768,eps=1e-4,momentum=0.1),
            nn.Conv2d(768, 896, 2, padding=1), # 7
            nn.ELU(),
            #nn.BatchNorm2d(896,eps=1e-4,momentum=0.1),
            nn.Conv2d(896, 896, 2, padding=1), # 8
            nn.ELU(),
            nn.MaxPool2d(2, 2), # 4
            nn.Dropout2d(0.3),
            # Block 5num_classes
            nn.Conv2d(896, 896, 1, padding=0), # 4
            nn.ELU(),
           # nn.BatchNorm2d(896,eps=1e-4,momentum=0.1),
            nn.Conv2d(896, 1024, 2, padding=1), # 5
            nn.ELU(),
            #nn.BatchNorm2d(1024,eps=1e-4,momentum=0.1),
            nn.Conv2d(1024, 1024, 2, padding=1), # 6
            nn.ELU(),
            #nn.BatchNorm2d(1024,eps=1e-4,momentum=0.1),
            nn.MaxPool2d(2, 2), # 3
            nn.Dropout2d(0.4),
            # Block 6
            nn.Conv2d(1024, 1024, 1, padding=0), # 3
            nn.ELU(),
            #nn.BatchNorm2d(1024,eps=1e-4,momentum=0.1),
            nn.Conv2d(1024, 1152, 2, padding=0), # 2
            nn.ELU(),
            #nn.BatchNorm2d(1152,eps=1e-4,momentum=0.1),
            nn.MaxPool2d(2, 2), # 1
            nn.Dropout2d(0.5),
            # Block 7
            nn.Conv2d(1152, 1152, 1, padding=0), #  1
            nn.ELU(),
            #nn.BatchNorm2d(1152,eps=1e-4,momentum=0.1),
            nn.Dropout2d(0.0),
            # Block 8
            nn.Conv2d(1152, num_classes, 1, padding=0) # 1
        )
 
    def forward(self, x):
        return self.main(x).view(x.size(0),-1)

model=CIFARClassifierELUPaper()

print(model)
    
print_num_params(model)

def ZCA_whitening(data):
    
    data=data.view(data.size(0),-1)
    X_norm=data/255
    X_norm.mean(axis=0).shape
    cov = np.cov(X_norm.cpu(), rowvar=True)   
    U,S,V = np.linalg.svd(cov)
    epsilon = 0.1
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm.cpu())   
    X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())   
    
    return X_ZCA_rescaled


def zca_prepare_batch(batch, device=None, **kwargs):
    
    
    data, target = batch
    zca_data = ZCA_whitening(data).reshape(data.shape)
    
    zca_data = Variable(torch.from_numpy(zca_data))
    
    zca_data = zca_data.to(device)

    target = target.to(device)

    return zca_data, target


model = model.cuda()

criterion = nn.CrossEntropyLoss()
lr = 0.01

optimizer = optim.SGD(model.parameters(),lr,momentum=0.9,weight_decay=0.0005,nesterov=True)

#lr_scheduler = ExponentialLR(optimizer, gamma=0.975)
lr_scheduler = MultiStepLR(optimizer, milestones=[70], gamma=0.1)

use_amp = True

#Load state dict
filename='checkpoint_testxxx.pth'
if os.path.isfile(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #print('lr scheduler is')
    #print(checkpoint['lr_scheduler'])
    print("=> loaded checkpoint ")



# Initialize Amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O2", num_losses=1)
   
    
batch = next(iter(train_loader))

batch = None
torch.cuda.empty_cache()

trainer = create_supervised_trainer(
    model, 
    optimizer, 
    criterion, 
    device="cuda",
    prepare_batch=zca_prepare_batch
)


#def output_transform(out):
#    return out['batchloss']

#RunningAverage(output_transform=output_transform).attach(trainer, "batchloss")

print('Start experiment')

trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())


resume_epoch = 69 # zero-based


def resume_training(engine):
    engine.state.iteration = resume_epoch * len(engine.state.dataloader)
    engine.state.epoch = resume_epoch
    print('Las iteraciones son')
    print(engine.state.iteration)
    print('El epoch actual es')
    print(engine.state.epoch)

#trainer.add_event_handler(Events.STARTED, resume_training)

metrics = {
    'Loss': Loss(criterion),
    'Accuracy': Accuracy(),
    'Precision': Precision(average=True),
    'Recall': Recall(average=True),
    'Top-5 Accuracy': TopKCategoricalAccuracy(k=5)
}

evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,prepare_batch=zca_prepare_batch, non_blocking=True)
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,prepare_batch=zca_prepare_batch, non_blocking=True)


cpe = CustomPeriodicEvent(n_epochs=3)
cpe.attach(trainer)


def run_evaluation(engine):
    train_evaluator.run(eval_train_loader)
    evaluator.run(test_loader)

def save_model_and_metrics(engine):
#    print('el num de epochs actual es')
#    print(trainer.state.epoch)
    epoch=engine.state.epoch
    if len(str(epoch))==1:
        epoch='00'+str(epoch)
    if len(str(epoch))==2:
        epoch='0'+str(epoch)
        

trainer.add_event_handler(cpe.Events.EPOCHS_3_STARTED, run_evaluation)
trainer.add_event_handler(Events.COMPLETED, run_evaluation)


trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

training_saver = ModelCheckpoint("checkpoint_190520",
                                filename_prefix='checkpoint',
                                save_interval=None,  # Save every 1000 iterations
                                n_saved=None,
                                atomic=True,
                                save_as_state_dict=True,
                                require_empty=False,
                                create_dir=True)
#Changed from Events.ITERATION_COMPLETED to Events.EPOCH_COMPLETED. EDIT: changed to EPOCH_STARTED every 10

trainer.add_event_handler(Events.EPOCH_STARTED(every=3), training_saver, 
{
                              "model": model,
                              "optimizer": optimizer,
                              "lr_scheduler": lr_scheduler
                          })
# Store the best model
def default_score_fn(engine):
    score = engine.state.metrics['Accuracy']
    return score


# Add early stopping
es_patience = 10
es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
#evaluator.add_event_handler(Events.COMPLETED, es_handler)
#setup_logger(es_handler._logger)

# Clear cuda cache between training/testing
def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
#train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)  

num_epochs =80

ProgressBar(persist=True).attach(trainer)

trainer.run(train_loader, max_epochs=num_epochs)

print('The results are')
print(train_evaluator.state.metrics) 
print(evaluator.state.metrics) 

# Dill routine
    
model_copy=dill.dumps(model)
torch.save(model_copy,'complete_model_final.pt')
torch.save(train_evaluator.state.metrics,'metrics_final.pt')
