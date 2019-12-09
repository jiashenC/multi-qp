import torch
from tqdm import tqdm, trange
import numpy as np
from parse_train_csv import get_images_from_tracks
from torch.utils.data import Dataset, DataLoader
from ImageDataset import ImageDataset
from pathlib import Path

def get_train_val_embeddings(triplet_model, train_tracks, val_tracks):
    train_images = get_images_from_tracks(train_tracks)
    val_images = get_images_from_tracks(val_tracks)

    split_index = len(train_images)
    total_images = train_images + val_images
    total_images_dataset = ImageDataset(total_images, Path("image_train"), (256, 256))
    total_images_dataloader = DataLoader(total_images_dataset, num_workers=4, batch_size=64, shuffle=False)

    total_embeddings = []
    with torch.no_grad():
        triplet_model = triplet_model.eval()
        for image_batch in tqdm(total_images_dataloader):
            image_batch = image_batch.cuda()
            batch_embeddings = triplet_model.getEmbedding(image_batch)
            total_embeddings.append(batch_embeddings.cpu())
    total_embeddings = torch.cat(total_embeddings, dim=0).numpy()
    val_embeddings = total_embeddings[split_index:, :]

    return total_embeddings, val_embeddings

def indices_to_vehicle_ids(matches, total_images, ID_to_image_dict, split_index=0):
    num_queries = matches.shape[0]
    num_database = matches.shape[1]
    id_matches = np.empty_like(matches, dtype=object)
    # expected_matches = []
    for query in trange(num_queries):
        for database in range(num_database):
            id_matches[query, database] = total_images[matches[query, database]]['vehicleID']
        # query_matches = ID_to_image_dict[total_images[split_index + query]['vehicleID']]
        # query_matches = np.array([item['vehicleID'] for item in query_matches])
        # expected_matches.append(query_matches)
    return id_matches