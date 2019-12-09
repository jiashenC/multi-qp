from tqdm import tqdm_notebook, trange, tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from TripletNet import TripletNet
from TripletDataset import TripletDataset
from torch.utils.data import Dataset, DataLoader
from parse_train_csv import get_train_val_tracks
from pathlib import Path

NUM_EPOCHS = 25
triplet_model = TripletNet().cuda()
criterion = nn.MarginRankingLoss(margin = 1)
optimizer = optim.Adam(triplet_model.parameters(), lr=0.001)

losses = []

train_tracks, val_tracks = get_train_val_tracks()
triplet_dataset = TripletDataset(train_tracks, Path('image_train'), (512, 512))
triplet_dataloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

for epoch in trange(NUM_EPOCHS):

    for triplet_batch in triplet_dataloader:
        anchors, positives, negatives = triplet_batch
        anchors, positives, negatives = anchors.cuda(), positives.cuda(), negatives.cuda()
        dist_a, dist_b, embedded_x, embedded_y, embedded_z = triplet_model(anchors, positives, negatives)
        target = torch.FloatTensor(dist_a.size()).fill_(1).cuda()
        loss_triplet = criterion(dist_a, dist_b, target)
        # loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet

        losses.append(loss.item())
        scheduler.step(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(triplet_model.state_dict(), "triplet_model.pth")