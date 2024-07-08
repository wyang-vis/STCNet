# --20200813   ma
# Ref: fast_blind_video_temporal_consistency
### python lib
import os, sys, math, random, glob, cv2, h5py, logging, random
import numpy as np

### torch lib
import torch
import torch.utils.data as data
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
### custom lib
import utils

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]

    pols[pols == 0] = -1  # polarity should be +1 / -1,这里没有问题

    tis = ts.astype(np.int)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)

    vals_right = pols * dts


    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    # print('voxel_grid',voxel_grid)

    return voxel_grid



class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def create_data_loader(data_set, opts, mode='train'):

    total_samples = opts.train_iters * opts.OPTIM.BATCH_SIZE


    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=4,
                             batch_size=opts.OPTIM.BATCH_SIZE, sampler=sampler, pin_memory=True,shuffle=False,drop_last=False)

    return data_loader





def train_dataset_in_binary(train_files_list,opt):
    '''
    创建用于训练的 dataset ，同时获取 divede-to-tensor 和 acc4zero-to-tensor 作为模型输入
    '''
    train_dataset = E2TensorByFrameDatasetBinary(train_files_list, opt)
    return train_dataset




class E2TensorByFrameDatasetBinary(data.Dataset):
    '''
    暂时仅针对灰度事，按照两帧之间的事件，融合 divide 和 acc4zero 的方法
    按照 从零累积灰度的方式，将灰度事件流转为tensor。每次只更新两帧frame之间的事件。
    '''
    def __init__(self, train_files_list, args):
        super(E2TensorByFrameDatasetBinary, self).__init__()
        self.args=args
        self.sequences_list = train_files_list
        logging.info('[%s] Total %d event sequences:' %
                         (self.__class__.__name__, len(self.sequences_list)))

        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)


    def __getitem__(self, index):
        file_item = self.sequences_list[index]
        with h5py.File(file_item, 'r') as f:
            blur_image_name_list = sorted(list(f['blur_images'].keys()))
            sharp_image_name_list = sorted(list(f['sharp_images'].keys()))
            event_image_name_list = sorted(list(f['event_frames'].keys()))
            blur_num_images = len(blur_image_name_list)
            DVS_stream_height = 720
            DVS_stream_width = 1280

            # print(blur_num_images)
            start_index = random.randint(0, blur_num_images - self.args.unrolling_len - 1)  # 起始图像索引,我觉得这里不用减1

            event_frame_list = np.zeros([self.args.unrolling_len * 6, DVS_stream_height, DVS_stream_width],
                                        dtype=np.float32)
            sharp_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                      dtype=np.float32)
            blur_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                     dtype=np.float32)
            # print('start_index', start_index)

            for t in range(self.args.unrolling_len):
                frame_index = t + start_index

                blur_img = np.asarray(f['blur_images'][blur_image_name_list[frame_index]])
                sharp_img = np.asarray(f['sharp_images'][sharp_image_name_list[frame_index]])
                event_frame = np.asarray(f['event_frames'][event_image_name_list[frame_index]])

                event_frame_list[t * 6:(t + 1) * 6, :, :] = event_frame
                blur_img_list[t * 3:(t + 1) * 3, :, :] = blur_img.transpose([2, 0, 1])
                sharp_img_list[t * 3:(t + 1) * 3, :, :] = sharp_img.transpose([2, 0, 1])

        input_img, input_event, target = utils.image_proess(blur_img_list, event_frame_list, sharp_img_list,
                                                            self.args.TRAINING.TRAIN_PS, self.args)
        data = (input_img, input_event, target)

        return data



class Train_Dataset_Binary_npz(data.Dataset):
    '''
    暂时仅针对灰度事，按照两帧之间的事件，融合 divide 和 acc4zero 的方法
    按照 从零累积灰度的方式，将灰度事件流转为tensor。每次只更新两帧frame之间的事件。
    '''
    def __init__(self, rgb_dir, args):
        super(Train_Dataset_Binary_npz, self).__init__()
        self.args=args
        self.blur_img_path=os.path.join(rgb_dir, 'blur')
        print(self.blur_img_path)
        self.event_img_path=os.path.join(rgb_dir, 'event_0.1_0.1')
        self.shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))##['GOPR0372_07_00', 'GOPR0372_07_01'
        print(inp_files_dirs)

        # event_files_dirs = sorted(os.listdir(event_img_path))
        #
        # tar_files_dirs = sorted(os.listdir(shapr_img_path))



        self.sequences_list = inp_files_dirs
        logging.info('[%s] Total %d event sequences:' %
                         (self.__class__.__name__, len(self.sequences_list)))

        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)


    def __getitem__(self, index):
        file_item = self.sequences_list[index]

        blur_image_name_list = sorted(glob.glob(os.path.join(self.blur_img_path,file_item, '*.png')))

        sharp_image_name_list = sorted(glob.glob(os.path.join(self.shapr_img_path,file_item, '*.png')))
        event_image_name_list = sorted(glob.glob(os.path.join(self.event_img_path,file_item, '*.npz')))
        blur_num_images = len(blur_image_name_list)
        DVS_stream_height = 720
        DVS_stream_width = 1280

        # print(blur_num_images)
        start_index = random.randint(0, blur_num_images - self.args.unrolling_len - 1)  # 起始图像索引,我觉得这里不用减1

        event_frame_list = np.zeros([self.args.unrolling_len * 6, DVS_stream_height, DVS_stream_width],
                                    dtype=np.float32)
        sharp_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                  dtype=np.float32)
        blur_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                 dtype=np.float32)


        for t in range(self.args.unrolling_len):
            frame_index = t + start_index

            blur_img = cv2.imread(blur_image_name_list[frame_index])
            blur_img = np.float32(blur_img) / 255.0

            sharp_img = cv2.imread(sharp_image_name_list[frame_index])
            sharp_img = np.float32(sharp_img) / 255.0

            event=np.load(event_image_name_list[frame_index])

            if len(event['t'])==0:
                event_div_tensor = np.zeros((self.args.num_bins, DVS_stream_height, DVS_stream_width))
            else:
                event_window = np.stack((event['t'],event['x'],event['y'],event['p']),axis=1)
                event_div_tensor = binary_events_to_voxel_grid(event_window,
                                                 num_bins=self.args.num_bins,
                                                 width=DVS_stream_width,
                                                 height=DVS_stream_height)
            event_frame=event_div_tensor


            event_frame_list[t * 6:(t + 1) * 6, :, :] = event_frame
            blur_img_list[t * 3:(t + 1) * 3, :, :] = blur_img.transpose([2, 0, 1])
            sharp_img_list[t * 3:(t + 1) * 3, :, :] = sharp_img.transpose([2, 0, 1])

        input_img, input_event, target = utils.image_proess(blur_img_list, event_frame_list, sharp_img_list,
                                                            self.args.TRAINING.TRAIN_PS, self.args)
        data = (input_img, input_event, target)

        return data



class E2TensorByFrameDatasetBinary_npy(data.Dataset):
    '''
    暂时仅针对灰度事，按照两帧之间的事件，融合 divide 和 acc4zero 的方法
    按照 从零累积灰度的方式，将灰度事件流转为tensor。每次只更新两帧frame之间的事件。
    '''
    def __init__(self, rgb_dir, args):
        super(E2TensorByFrameDatasetBinary_npy, self).__init__()
        self.args=args
        self.blur_img_path=os.path.join(rgb_dir, 'blur')
        print(self.blur_img_path)
        self.event_img_path=os.path.join(rgb_dir, 'event_0.1_0.1')
        self.shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))##['GOPR0372_07_00', 'GOPR0372_07_01'
        print(inp_files_dirs)

        # event_files_dirs = sorted(os.listdir(event_img_path))
        #
        # tar_files_dirs = sorted(os.listdir(shapr_img_path))



        self.sequences_list = inp_files_dirs
        logging.info('[%s] Total %d event sequences:' %
                         (self.__class__.__name__, len(self.sequences_list)))

        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)


    def __getitem__(self, index):
        file_item = self.sequences_list[index]

        blur_image_name_list = sorted(glob.glob(os.path.join(self.blur_img_path,file_item, '*.png')))

        sharp_image_name_list = sorted(glob.glob(os.path.join(self.shapr_img_path,file_item, '*.png')))
        event_image_name_list = sorted(glob.glob(os.path.join(self.event_img_path,file_item, '*.npy')))
        blur_num_images = len(blur_image_name_list)
        DVS_stream_height = 720
        DVS_stream_width = 1280

        # print(blur_num_images)
        start_index = random.randint(0, blur_num_images - self.args.unrolling_len - 1)  # 起始图像索引,我觉得这里不用减1

        event_frame_list = np.zeros([self.args.unrolling_len * 6, DVS_stream_height, DVS_stream_width],
                                    dtype=np.float32)
        sharp_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                  dtype=np.float32)
        blur_img_list = np.zeros([self.args.unrolling_len * 3, DVS_stream_height, DVS_stream_width],
                                 dtype=np.float32)


        for t in range(self.args.unrolling_len):
            frame_index = t + start_index

            blur_img = cv2.imread(blur_image_name_list[frame_index])
            blur_img = np.float32(blur_img) / 255.0

            sharp_img = cv2.imread(sharp_image_name_list[frame_index])
            sharp_img = np.float32(sharp_img) / 255.0

            event=np.load(event_image_name_list[frame_index])


            event_frame = np.float32(event)

            event_frame_list[t * 6:(t + 1) * 6, :, :] = event_frame
            blur_img_list[t * 3:(t + 1) * 3, :, :] = blur_img.transpose([2, 0, 1])
            sharp_img_list[t * 3:(t + 1) * 3, :, :] = sharp_img.transpose([2, 0, 1])

        input_img, input_event, target = utils.image_proess(blur_img_list, event_frame_list, sharp_img_list,
                                                            self.args.TRAINING.TRAIN_PS, self.args)
        data = (input_img, input_event, target)

        return data





class ValDatasetFromH5PYBinaryByFrames:

    def __init__(self, h5py_object, args):
        self.args=args
        self.f =h5py.File(h5py_object,'r')
        self.blur_image_name_list = sorted(list(self.f ['blur_images'].keys()))
        self.sharp_image_name_list = sorted(list(self.f ['sharp_images'].keys()))
        self.event_image_name_list = sorted(list(self.f ['event_frames'].keys()))
        self.blur_num_images = len(self.blur_image_name_list)
        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.blur_num_images - 2:
            event_frame_list = np.zeros([self.args.unrolling_len * 6, self.DVS_stream_height, self.DVS_stream_width],
                                        dtype=np.float32)
            sharp_img_list = np.zeros([self.args.unrolling_len * 3, self.DVS_stream_height, self.DVS_stream_width],
                                      dtype=np.float32)
            blur_img_list = np.zeros([self.args.unrolling_len * 3, self.DVS_stream_height, self.DVS_stream_width],
                                     dtype=np.float32)
            for t in range(self.args.unrolling_len):
                frame_index = t + self.index

                blur_img = np.asarray(self.f['blur_images'][self.blur_image_name_list[frame_index]])
                sharp_img = np.asarray(self.f['sharp_images'][self.sharp_image_name_list[frame_index]])
                event_frame = np.asarray(self.f['event_frames'][self.event_image_name_list[frame_index]])


                event_frame_list[t * 6:(t + 1) * 6, :, :] = event_frame
                blur_img_list[t * 3:(t + 1) * 3, :, :] = blur_img.transpose([2, 0, 1])
                sharp_img_list[t * 3:(t + 1) * 3, :, :] = sharp_img.transpose([2, 0, 1])


            input_img, input_event, target = utils.image_proess_val(blur_img_list, event_frame_list, sharp_img_list, self.args)
            self.index += 1
            data = (input_img, input_event, target)
            return data
        else:
            raise StopIteration

