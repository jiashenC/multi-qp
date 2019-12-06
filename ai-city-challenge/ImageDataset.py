from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, images, root_folder, image_size):
        self.images = images
        self.root_folder = root_folder
        self.image_size = image_size
    
    def __len__(self):
        return len(self.images)
    
    def applyTransforms(self, image):
        image_resized = TF.resize(image, self.image_size)
        image_tensor = TF.to_tensor(image_resized)
        return image_tensor
    
    def __getitem__(self, idx):
        image = Image.open(self.root_folder/self.images[idx]["imageName"])
        image = self.applyTransforms(image)
        return image