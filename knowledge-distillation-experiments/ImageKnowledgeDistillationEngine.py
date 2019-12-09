import time
import numpy as np
import os.path as osp
import datetime
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchreid.engine import Engine
from torchreid import metrics
from torchreid.utils import (
    AverageMeter, re_ranking, save_checkpoint, visualize_ranked_results
)
from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
# from torchreid.losses import DeepSupervision
from torchreid.losses import CrossEntropyLoss
from KnowledgeDistillationLoss import KnowledgeDistillationLoss

class ImageKnowledgeDistillationEngine(Engine):
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        teacher_model,
        alpha,
        temperature,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageKnowledgeDistillationEngine, self
              ).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = teacher_model

        self.criterion = KnowledgeDistillationLoss(self.temperature, self.alpha)

    def train(
        self,
        epoch,
        max_epoch,
        writer,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None
    ):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(imgs)
            loss = self.criterion(teacher_outputs, outputs, pids)
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                    num_batches - (batch_idx+1) + (max_epoch -
                                                   (epoch+1)) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accs,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch*num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Data', data_time.avg, n_iter)
                writer.add_scalar('Train/Loss', losses.avg, n_iter)
                writer.add_scalar('Train/Acc', accs.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()