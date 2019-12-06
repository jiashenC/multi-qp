import numpy as np
from tqdm import trange

def get_precision(ids, index, max_num_elements = 1000):
    
    required_id = ids[index, 0]

    # num_possible_images = ID_to_image_dict[id_matches[0, 0]]
    # same_query_indices = np.argwhere(id_matches[:, 0] == id_matches[index, 0])
    non_query_indices = np.argwhere(ids[index, :max_num_elements] != ids[index, 0])

    true_positives = np.cumsum(ids[index, :max_num_elements+1] == required_id)
    true_positives[non_query_indices] = 0
    true_positives = true_positives[1:]
    # print(true_positives)
    num_values = true_positives[-1]
    # num_values = np.count_nonzero(id_matches[index, :] == id_matches[index, 0])

    true_positives = true_positives/(np.arange(max_num_elements)+1)
    # print(true_positives)
    true_positives = np.sum(true_positives)
    # false_positives = np.cumsum(np.sum(id_matches[non_query_indices, :max_num_elements] == required_id, axis=0))
    # num_matches = np.cumsum(ids[index] == required_id) - 1

    if num_values == 0:
        return 0
    average_precision = true_positives*1.0/num_values

    # print(true_positives)
    # print(false_positives)
        
    return average_precision

def get_average_precision(id_matches):
    ap_list = []

    for query in trange(id_matches.shape[0]):
        average_precision = get_precision(id_matches, query, max_num_elements=100)
        ap_list.append(average_precision)

    return np.mean(ap_list)