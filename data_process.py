import numpy as np
import glob
import os
from scipy import spatial
import pickle

SAVE_DIR = './processed_data'
os.makedirs(SAVE_DIR, exist_ok=True)

# Please change this to your location
data_root = './data/CWL'

history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 120  # maximum number of observed objects is 70
neighbor_distance = 10  # meter

# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = 10 + 1  # we add mark "1" to the end of each row to indicate that this row exists

# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
    '''
    Read raw data from files and return a dictionary:
        {frame_id:
            {object_id:
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
            }
        }
    '''
    with open(pra_file_path, 'r') as reader:
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)
        now_dict = {}
        for row in content:
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1]] = row  # [2:]
            now_dict[row[0]] = n_dict
    return now_dict

def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
    visible_object_id_list = list(pra_now_dict[pra_observed_last].keys())  # object_id appears at the last observed frame
    num_visible_object = len(visible_object_id_list)  # number of current observed objects

    # compute the mean values of x and y for zero-centralization.
    visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
    xy = visible_object_value[:, 3:5].astype(float)
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[3:5] = m_xy

    # compute distance between any pair of two objects
    dist_xy = spatial.distance.cdist(xy, xy)
    # if their distance is less than $neighbor_distance, we regard them are neighbors.
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy < neighbor_distance).astype(int)

    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)

    object_feature_list = []
    for frame_ind in range(pra_start_ind, pra_end_ind):
        # we add mark "1" to the end of each row to indicate that this row exists
        # -mean_xy is used to zero_centralize data
        now_frame_feature_dict = {
            obj_id: (list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [1]
                     if obj_id in visible_object_id_list
                     else list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [0])
            for obj_id in pra_now_dict[frame_ind]
        }
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
        now_frame_feature = np.array(
            [now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension))
             for vis_id in visible_object_id_list + non_visible_object_id_list]
        )
        object_feature_list.append(now_frame_feature)

    # object_feature_list has shape of (frame#, object#, 11)
    object_feature_list = np.array(object_feature_list)

    # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
    object_frame_feature = np.zeros((max_num_object, pra_end_ind - pra_start_ind, total_feature_dimension))

    object_frame_feature[:num_visible_object + num_non_visible_object] = np.transpose(object_feature_list, (1, 0, 2))
    return object_frame_feature, neighbor_matrix, m_xy

def generate_train_data(pra_file_path):
    '''
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feture: (N, C, T, V)
            N is the number of training data
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects.
    '''
    now_dict = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    for start_ind in frame_id_set[:-total_frames + 1]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames)
        observed_last = start_ind + history_frames - 1
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    return all_feature_list, all_adjacency_list, all_mean_list

def generate_test_data(pra_file_path):
    now_dict = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    # get all start frame id
    start_frame_id_list = frame_id_set[::history_frames]
    for start_ind in start_frame_id_list:
        start_ind = int(start_ind)
        end_ind = int(start_ind + history_frames)
        observed_last = start_ind + history_frames - 1
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    return all_feature_list, all_adjacency_list, all_mean_list

def generate_data(pra_file_path_list, pra_is_train=True, train_data_size=None, custom_save_name=None):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    for file_path in pra_file_path_list:
        if pra_is_train:
            now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
        else:
            now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
        all_data.extend(now_data)
        all_adjacency.extend(now_adjacency)
        all_mean_xy.extend(now_mean_xy)

    all_data = np.array(all_data)        # (N, C, T, V)
    all_adjacency = np.array(all_adjacency)  # (N, V, V)
    all_mean_xy = np.array(all_mean_xy)  # (N, 2)

    print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

    # save training_data and trainjing_adjacency into a file.
    if custom_save_name is not None:
        save_path = os.path.join(SAVE_DIR, custom_save_name)
    elif pra_is_train and train_data_size is not None:
        save_path = os.path.join(SAVE_DIR, f'train_data_{train_data_size}.pkl')
    else:
        fname = 'train_data.pkl' if pra_is_train else 'test_data.pkl'
        save_path = os.path.join(SAVE_DIR, fname)

    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy], writer)

if __name__ == '__main__':
    train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    # ===== 5 users (1–10, 11–20, 21–30, 31–40, 41–53) + ALL (53) =====
    user1_files = train_file_path_list[0:10]
    user2_files = train_file_path_list[10:20]
    user3_files = train_file_path_list[20:30]
    user4_files = train_file_path_list[30:40]
    user5_files = train_file_path_list[40:53]  # 13 files
    all_files   = train_file_path_list         # 53 files

    print('Generating Training Data for USER 1 (10 files).')
    generate_data(user1_files, pra_is_train=True, custom_save_name='train_data_user1.pkl')

    print('Generating Training Data for USER 2 (10 files).')
    generate_data(user2_files, pra_is_train=True, custom_save_name='train_data_user2.pkl')

    print('Generating Training Data for USER 3 (10 files).')
    generate_data(user3_files, pra_is_train=True, custom_save_name='train_data_user3.pkl')

    print('Generating Training Data for USER 4 (10 files).')
    generate_data(user4_files, pra_is_train=True, custom_save_name='train_data_user4.pkl')

    print('Generating Training Data for USER 5 (13 files).')
    generate_data(user5_files, pra_is_train=True, custom_save_name='train_data_user5.pkl')

    print('Generating Training Data for ALL (53 files).')
    generate_data(all_files, pra_is_train=True, custom_save_name='train_data_all.pkl')

    print('Generating Testing Data.')
    generate_data(test_file_path_list, pra_is_train=False)
    