import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import os
import cv2
import glob
import time
import tqdm
import tensorboardX
from functools import partial

from apex import amp
import matplotlib.pyplot as plt

import gzip

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def make_tensor(data_xy):
    data_x, data_y = data_xy
    data_x = np.array(data_x, dtype=np.float32)
    data_y = np.array(data_y, dtype=np.int32)
    return data_x, data_y

def load_pickle(f):
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def load_data(data_file):
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set) # (50000, 784) (50000,)
    valid_set_x, valid_set_y = make_tensor(valid_set) # (10000, 784) (10000,)
    test_set_x, test_set_y = make_tensor(test_set) # (10000, 784) (10000,)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


def loss(H1, H2, outdim_size, use_all_singular_values=False):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9

    H1, H2 = H1.t(), H2.t()

    o1 = o2 = H1.size(0)

    m = H1.size(1)

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=H1.device)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
    [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

    # Added to increase stability
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    if use_all_singular_values:
        # all singular values are used to calculate the correlation
        tmp = torch.matmul(Tval.t(), Tval)
        corr = torch.trace(torch.sqrt(tmp))
        # assert torch.isnan(corr).item() == 0
    else:
        # just the top outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(H1.device))
        U = U.topk(outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))

    return - corr

class MLP(nn.Module):
    def __init__(self, chan):
        super().__init__()
        layers = nn.ModuleList([])
        cin = chan[0]
        for cout in chan[1:-1]:
            layers.append(nn.Linear(cin, cout))
            layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(True))
            cin = cout
        layers.append(nn.Linear(cin, chan[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DeepCCA(nn.Module):
    def __init__(self, chan1, chan2):
        super().__init__()

        self.model1 = MLP(chan1)
        self.model2 = MLP(chan2)
        
    def forward(self, x1, x2):
        # feature * batch_size
        z1 = self.model1(x1)
        z2 = self.model2(x2)

        return z1, z2

class NoisyMNIST(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y
    
    def __getitem__(self, i):
        data = {
            'x1': self.X1[i],
            'x2': self.X2[i],
            'y': self.y[i]
        }

        return data
    
    def __len__(self):
        #return 100
        return self.X1.shape[0]


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 objective, # loss function
                 optimizer=None, # optimizer
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 opt_level='O0', # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=3, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.mute = mute
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.opt_level = opt_level
        self.best_mode = best_mode
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        if isinstance(self.objective, nn.Module):
            self.objective.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level, verbosity=0)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, "log.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}_best.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Train from scratch")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log("[INFO] Best checkpoint not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # extra
        self.Z1 = []
        self.Z2 = []

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args):
        if self.local_rank == 0:
            if not self.mute: 
                print(*args)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)    

    ### ------------------------------	

    def train_step(self, data):
        x1, x2 = data['x1'], data['x2']
        z1, z2 = self.model(x1, x2)
        loss = self.objective(z1, z2)
        return (z1, z2), None, loss

    def eval_step(self, data):
        (z1, z2), truths, loss = self.train_step(data)
        self.Z1.append(z1) # [B, fout]
        self.Z2.append(z2)
        return (z1, z2), truths, loss

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(True if epoch == max_epochs else False) # save full at last epoch
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        if os.path.exists(self.best_path):
            self.load_checkpoint(self.best_path)
        else:
            self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.Z1 = []
        self.Z2 = []
        self.evaluate_one_epoch(loader)
        self.Z1 = torch.cat(self.Z1, dim=0).detach().cpu().numpy()
        self.Z2 = torch.cat(self.Z2, dim=0).detach().cpu().numpy()
        self.use_tensorboardX = use_tensorboardX

        # return transformed results !
        return self.Z1, self.Z2

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")

        total_loss = []
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)
            preds, truths, loss = self.train_step(data)

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss.append(loss.item())
            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={total_loss[-1]:.4f}, lr={self.optimizer.param_groups[0]['lr']}")
                else:
                    pbar.set_description(f'loss={total_loss[-1]:.4f}')
                pbar.update(loader.batch_size * self.world_size)

        average_loss = np.mean(total_loss)
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            for metric in self.metrics:
                self.log(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, average_loss={average_loss:.4f}")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = []
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                data = self.prepare_data(data)
                preds, truths, loss = self.eval_step(data)
                
                total_loss.append(loss.item())            
                if self.local_rank == 0:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                    pbar.set_description(f'loss={total_loss[-1]:.4f}')
                    pbar.update(loader.batch_size * self.world_size)

        average_loss = np.mean(total_loss)
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished, average_loss={average_loss:.4f}")

    def save_checkpoint(self, full=False):
        file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

        self.stats["checkpoints"].append(file_path)

        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = self.stats["checkpoints"].pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
            'model': self.model.state_dict(),
        }

        if full:
            state['amp'] = amp.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        torch.save(state, file_path)
        
        if len(self.stats["results"]) > 0:
            if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                self.stats["best_result"] = self.stats["results"][-1]
                torch.save(state, self.best_path)
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint_dict['model'])
        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch'] + 1
        if 'optimizer' in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'lr_scheduler' in checkpoint_dict:
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        if 'amp' in checkpoint_dict:
            amp.load_state_dict(checkpoint_dict['amp'])

        self.log("[INFO] Checkpoint Loaded Successfully.")

############################## main ###############################

# load data
# Note: y1 == y2, two views are from the same number.
X1_train, y1_train, X1_valid, y1_valid, X1_test, y1_test = load_data('../data/mnist/noisymnist_view1.gz')
X2_train, y2_train, X2_valid, y2_valid, X2_test, y2_test = load_data('../data/mnist/noisymnist_view2.gz')

# model
model = DeepCCA([784, 1024, 1024, 1024, 10], [784, 1024, 1024, 1024, 10])

# loss
objective = partial(loss, outdim_size=10, use_all_singular_values=False)

# data loader
train_dataset = NoisyMNIST(X1_train, X2_train, y1_train)
valid_dataset = NoisyMNIST(X1_valid, X2_valid, y1_valid)
test_dataset = NoisyMNIST(X1_test, X2_test, y1_test)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# trainer
seed_everything(2)
trainer = Trainer('DCCA', model, objective, optimizer=optimizer, use_checkpoint='scratch')

trainer.train(train_loader, valid_loader, 20)

train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False) # ordered with y_train
Z1_train, Z2_train = trainer.evaluate(train_loader2)
Z1_valid, Z2_valid = trainer.evaluate(valid_loader)
Z1_test, Z2_test = trainer.evaluate(test_loader)

# SVM classify
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

clf = LinearSVC(C=0.01, dual=False)
clf.fit(Z1_train, y1_train)

train_acc = accuracy_score(y1_train, clf.predict(Z1_train))
valid_acc = accuracy_score(y1_valid, clf.predict(Z1_valid))
test_acc = accuracy_score(y1_test, clf.predict(Z1_test))

print(train_acc, valid_acc, test_acc)

# T-SNE of Z1
tsne = TSNE()
Z1_tsne = tsne.fit_transform(Z1_test)
plt.scatter(Z1_tsne[:, 0], Z1_tsne[:, 1], c=y1_test, cmap='tab10')
plt.show()

