import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

# import your model class
# import ...
from Yandhi import Yandhi, Segmentation

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1():
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2():
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Yandhi'
    team_number = 44
    round_number = 3
    team_member = ['vish', 'ben', 'tony']
    contact_email = 'brs426@nyu.edu'

    def __init__(self, fl_path =):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #
        
        self.model = Yandhi.Yandhi()
        self.model.load_state_dict(torch.load(model_file_1))
        self.model.cuda()
        
        class_label = 0
        class_dict = dict()
        self.reverse_class_dict = []
        for i in range(0, 800, 50):
            for j in range(0, 800, 50):
                class_dict[(i, j)] = class_label
                class_label += 1
                self.reverse_class_dict.append((i, j))
        

    def get_bounding_boxes(self, samples):
        self.model.eval()
        bb_samples = []
        
        y_hat, y_count, segmentation = self.model(samples)
        
        if torch.argmax(y_count).item() > 15:
            result = torch.topk(y_hat, k = 6 + torch.argmax(y_count).item())
            pred_ids = result.indices
        else:
            result = torch.topk(y_hat, k = torch.argmax(y_count).item())
            pred_ids = result.indices

        bounding_boxes = []
        for idx in pred_ids[0]:
            bin_x, bin_y = self.reverse_class_dict[idx.item()]

            xs = torch.Tensor([bin_x, bin_x, bin_x + 50, bin_x + 50]).double()
            ys = torch.Tensor([bin_y+16, bin_y+36, bin_y+16, bin_y+36]).double()

            xs = xs - 400
            ys = 800 - ys # right-side up
            ys = ys - 400

            xs /= 10.
            ys /= 10.

            coords = torch.stack((xs, ys))
            bounding_boxes.append(coords)

        bounding_boxes = torch.stack(bounding_boxes).double().cuda()
        bb_samples.append(bounding_boxes)
        bb_samples = tuple(bb_samples)
        return bb_samples

 
    def reconstruct_from_bins(self, bins, block_size):
        road_map = torch.zeros((800, 800))
        idx = 0
        for x in range(0, 800, block_size):
            for y in range(0, 800, block_size):
                road_map[x:x+block_size, y:y+block_size] = bins[idx].item()
                idx += 1
        return road_map > 0.4

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        road_maps = []
        self.model.eval()
        y_hat, y_count, segmentation = self.model(samples)
        for seg in segmentation:
            road_map = self.reconstruct_from_bins(seg, 5)
            road_maps.append(road_map)
        
        road_maps = torch.stack(road_maps).cuda()
        return road_maps
