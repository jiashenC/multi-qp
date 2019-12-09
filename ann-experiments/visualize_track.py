import math
import matplotlib.pyplot as plt
from pathlib import Path

def show_images_in_track(track):
    img_folder_path = Path('image_train')
    num_images = len(track)
    num_cols = 5
    num_rows = math.ceil(1.0*num_images/num_cols)
    if num_rows == 1:
        num_cols = num_images
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10,10))

    row = 0
    col = 0
    for i, img_name in enumerate(track):
        img = plt.imread(img_folder_path/img_name)
        if num_rows == 1:
            current_axis = axs[col]
        else:
            current_axis = axs[row, col]
        current_axis.axis("off")
        current_axis.set_title(img_name)
        current_axis.imshow(img)

        col += 1

        if col == num_cols:
            col = 0
            row += 1
    
    if num_rows*num_cols > num_images:
        remaining_axes = num_rows*num_cols - num_images 
        for _ in range(remaining_axes):
            current_axis = axs[row, col]
            current_axis.axis("off")
            col += 1
            if col == num_cols:
                col = 0
                row += 1
    
    plt.show()