from TripletNet import TripletNet
import torch
from get_train_val_embeddings import get_train_val_embeddings, indices_to_vehicle_ids
from sklearn.metrics import pairwise_distances, average_precision_score
from parse_train_csv import readAICityChallengeData, get_train_val_tracks, get_images_from_tracks
import numpy as np
from metrics import get_average_precision

triplet_model = TripletNet().cuda()

triplet_model.load_state_dict(torch.load("triplet_model.pth"))
train_tracks, val_tracks = get_train_val_tracks()
total_embeddings, val_embeddings = get_train_val_embeddings(triplet_model, train_tracks, val_tracks)
distances = pairwise_distances(val_embeddings, total_embeddings)

best_matches = np.argsort(distances, axis=1)

train_images = get_images_from_tracks(train_tracks)
val_images = get_images_from_tracks(val_tracks)

split_index = len(train_images)
total_images = train_images + val_images
_, ID_to_image_dict = readAICityChallengeData()
id_matches = indices_to_vehicle_ids(best_matches, total_images, ID_to_image_dict, split_index)

print(get_average_precision(id_matches))