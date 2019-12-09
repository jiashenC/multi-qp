from annoy import AnnoyIndex
from TripletNet import TripletNet
from get_train_val_embeddings import get_train_val_embeddings, indices_to_vehicle_ids
from sklearn.metrics import pairwise_distances, average_precision_score
from parse_train_csv import readAICityChallengeData, get_train_val_tracks, get_images_from_tracks
import numpy as np
import torch
from metrics import get_average_precision

triplet_model = TripletNet().cuda()
triplet_model.load_state_dict(torch.load("triplet_model.pth"))
train_tracks, val_tracks = get_train_val_tracks()
total_embeddings, val_embeddings = get_train_val_embeddings(triplet_model, train_tracks, val_tracks)

f = total_embeddings.shape[1]
t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
for i in range(total_embeddings.shape[0]):
    v = total_embeddings[i]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

annoy_image_matches = []
for query in range(val_embeddings.shape[0]):
    annoy_matches = t.get_nns_by_vector(val_embeddings[query], 101)
    annoy_image_matches.append(np.array(annoy_matches))

annoy_image_matches = np.stack(annoy_image_matches)

train_images = get_images_from_tracks(train_tracks)
val_images = get_images_from_tracks(val_tracks)

split_index = len(train_images)
total_images = train_images + val_images
_, ID_to_image_dict = readAICityChallengeData()

annoy_id_matches = indices_to_vehicle_ids(annoy_image_matches, total_images, ID_to_image_dict, split_index)

print()
print(get_average_precision(annoy_id_matches))