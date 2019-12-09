from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.model_selection import train_test_split

dataset_path = Path('.')

def readAICityChallengeData():
    ID_to_image_dict = defaultdict(lambda: [])
    with open(dataset_path/'train_label.xml','r') as f:
        ef = ET.fromstring(f.read())
        items = ef.getchildren()
        image_to_attribute = list(items[0].iter('Item'))
        image_to_attribute = {item.attrib["imageName"]:item.attrib for item in image_to_attribute}
        for key, item in image_to_attribute.items():
            ID_to_image_dict[item["vehicleID"]].append(item)
    
    return image_to_attribute, ID_to_image_dict

def get_train_tasks():
    with open(dataset_path/'train_track.txt', "r") as fo:
        lines = fo.readlines()
    lines = [line.split(' ')[:-1] for line in lines]
    return lines

def get_images(filename):
    with open(dataset_path/filename, "r") as fo:
        lines = fo.readlines()
    lines = [line[:-1] for line in lines]
    return lines

def get_images_from_tracks(tracks):
    image_to_attribute, ID_to_image_dict = readAICityChallengeData()
    images = [image_to_attribute[image] for track in tracks for image in track]
    return images

def get_train_val_tracks():
    train_tracks_total = get_train_tasks()

    # Filter those tracks which have less than 2 images
    train_tracks_total = [track for track in train_tracks_total if len(track) >=2]

    train_tracks, val_tracks = train_test_split(train_tracks_total, test_size=0.10, random_state=42)
    print("Training Size - {}".format(len(train_tracks)))
    print("Validation Size - {}".format(len(val_tracks)))

    return train_tracks, val_tracks