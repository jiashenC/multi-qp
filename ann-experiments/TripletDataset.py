from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def create_triplet_task(positive_track, negative_track):

    # positive_track, negative_track = np.random.choice(train_tracks, size=2, replace=False)
    anchor_image, positive_image = np.random.choice(positive_track, size=2, replace=False)
    negative_image = np.random.choice(negative_track)

    return anchor_image, positive_image, negative_image

def visualize_triplet_task(anchor_image, positive_track, negative_image):
    img_folder_path = Path('image_train')
    fig, axs = plt.subplots(1, 3, figsize=(10,10), gridspec_kw={'hspace': 0.2})
    images = [anchor_image, positive_track, negative_image]
    title = ["Anchor", "Positive", "Negative"]
    row = 0
    col = 0
    for i, img_name in enumerate(images):
        img = plt.imread(img_folder_path/img_name)
        axs[i].axis("off")
        axs[i].set_title(title[i] + " " + str(img.shape))
        axs[i].imshow(img)
    plt.show()

# visualize_triplet_task(*create_triplet_task(*np.random.choice(train_tracks_total, size=2, replace=False)))

class TripletDataset(Dataset):

    def __init__(self, tracks, image_path, image_size):
        self.tracks = tracks
        self.image_path = image_path
        self.image_size = image_size
    
    def __len__(self):
        return len(self.tracks)
    
    def applyTransforms(self, image):
        image_resized = TF.resize(image, self.image_size)
        image_tensor = TF.to_tensor(image_resized)
        return image_tensor
    
    def __getitem__(self, idx):
        positive_track = self.tracks[idx]
        negative_track_index = (idx + np.random.randint(1, len(self.tracks))) % len(self.tracks)
        assert idx != negative_track_index
        negative_track = self.tracks[negative_track_index]
        anchor_image, positive_image, negative_image = create_triplet_task(positive_track, negative_track)
        images = [anchor_image, positive_image, negative_image]
        images = [Image.open(self.image_path/image) for image in images]
        images = [self.applyTransforms(image) for image in images]
        anchor_image, positive_image, negative_image = images
        return anchor_image, positive_image, negative_image