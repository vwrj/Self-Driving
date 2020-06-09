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

    def __init__(self, model_paths = ['front_left.pt', 'front.pt', 'front_right.pt', 'back_left.pt', 'back.pt', 'back_right.pt'], segmentation_path = 'all_six_images_classify_count_better.pt'):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #

        self.fl_model = Yandhi(count_dim=28)
        self.f_model = Yandhi(count_dim=28)
        self.fr_model = Yandhi(count_dim=60)
        self.bl_model = Yandhi(count_dim=60)
        self.b_model = Yandhi(count_dim=60)
        self.br_model = Yandhi(count_dim=60)
        self.segmentation_model = Segmentation()

        self.fl_model.load_state_dict(torch.load(model_paths[0]))
        self.f_model.load_state_dict(torch.load(model_paths[1]))
        self.fr_model.load_state_dict(torch.load(model_paths[2]))
        self.bl_model.load_state_dict(torch.load(model_paths[3]))
        self.b_model.load_state_dict(torch.load(model_paths[4]))
        self.br_model.load_state_dict(torch.load(model_paths[5]))
        self.segmentation_model.load_state_dict(torch.load(segmentation_path))
        
        self.models = [self.fl_model, self.f_model, self.fr_model, self.bl_model, self.b_model, self.br_model, self.segmentation_model]
        for x in self.models:
            x.cuda()
            x.eval()
        
    def get_bounding_boxes(self, samples):
        
        fl_vehicle, _ = fl_model(sample[0][0][:, 130:, :].unsqueeze(0))
        f_vehicle, _ = f_model(sample[0][1][:, 130:, :].unsqueeze(0))
        fr_vehicle, _ = fr_model(sample[0][2][:, 120:, :].unsqueeze(0))
        bl_vehicle, _ = bl_model(sample[0][3][:, 130:, :].unsqueeze(0))
        b_vehicle, _ = b_model(sample[0][4][:, 130:, :].unsqueeze(0))
        br_vehicle, _ = br_model(sample[0][5][:, 120:, :].unsqueeze(0))

        fl_pred_map = torch.sigmoid(fl_vehicle[0])
        f_pred_map = torch.sigmoid(f_vehicle[0])
        fr_pred_map = torch.sigmoid(fr_vehicle[0])
        bl_pred_map = torch.sigmoid(bl_vehicle[0])
        b_pred_map = torch.sigmoid(b_vehicle[0])
        br_pred_map = torch.sigmoid(br_vehicle[0])

        reconstruct_fl_map = reconstruct_from_bins(fl_pred_map, 0.35).cpu()
        reconstruct_f_map = reconstruct_from_bins(f_pred_map, 0.4).cpu()
        reconstruct_fr_map = reconstruct_from_bins(fr_pred_map, 0.2).cpu()
        reconstruct_bl_map = reconstruct_from_bins(bl_pred_map, 0.3).cpu()
        reconstruct_b_map = reconstruct_from_bins(b_pred_map, 0.35).cpu()
        reconstruct_br_map = reconstruct_from_bins(br_pred_map, 0.6).cpu()

        reconstruct_front_map = reconstruct_fl_map + reconstruct_f_map + reconstruct_fr_map
        reconstruct_back_map = reconstruct_bl_map + reconstruct_b_map + reconstruct_br_map

        front_bboxes = get_bboxes(reconstruct_front_map)
        back_bboxes = get_bboxes(reconstruct_back_map)
        combined_bboxes = front_bboxes + back_bboxes

        if len(combined_bboxes) == 0:
            return tuple([])  
            
        
        return tuple(combined_bboxes)
        
        
    def reconstruct_from_bins(self, bins, threshold):
        road_map = torch.zeros((800, 800))
        idx = 0
        for x in range(0, 800, VERT_BLOCK_SIZE):
            for y in range(0, 800, HORIZ_BLOCK_SIZE):
                road_map[x:x+VERT_BLOCK_SIZE, y:y+HORIZ_BLOCK_SIZE] = bins[idx]
                idx += 1
        return road_map > threshold

    def faster_reconstruct_from_bins(self, bins, block_size, threshold):
        downsampled_roadmap = bins.reshape(1, 1, 800 // block_size, 800 // block_size)
        up = nn.Upsample(size=(800, 800), mode='nearest')
        road_map = up(downsampled_roadmap)
        return road_map > threshold

    def go(self, direction, top_left, width, height, bb_map):
        threshold = 0.0
        r, c = top_left
        delta = 10
        if direction == 'right':
            c = c + width 
            # now we're at the top-right coordinate. 
            while c + delta < 800:
                block = bb_map[r:r+height, c:c+delta]
                score = torch.sum(block).item()
                if score > threshold * height * delta:
                    c = c + delta
                    width = width + delta
                else:
                    break
                    
            return top_left, width, height
                    
        elif direction == 'left':
            # At top-left coordinate. 
            while c - delta > 0:
                block = bb_map[r:r+height, c-delta:c]
                score = torch.sum(block).item()
                if score > threshold * height * delta:
                    c = c - delta
                    width = width + delta
                else:
                    break
            
            return (r, c), width, height
        
        elif direction == 'up':
            # At top_left coordinate. 
            while r - delta > 0:
                block = bb_map[r-delta:r, c:c+width]
                score = torch.sum(block).item()
                if score > threshold * width * delta:
                    r = r - delta
                    height = height + delta
                else:
                    break
        
            return (r, c), width, height
    
        elif direction == 'down':
            r = r + delta
            # At bottom_left coordinate. 
            while r + delta < 800:
                block = bb_map[r:r+delta, c:c+width]
                score = torch.sum(block).item()
                if score > threshold * width * delta:
                    r = r + delta
                    height = height + delta
                else:
                    break
        
            return top_left, width, height
        
    def get_bboxes(recon_map):
        bb_map = recon_map.clone()

        score_threshold = 0
        bboxes = []
        for r in range(0, 800, 10):
            for c in range(0, 800, 10):

                top_left = (r, c)
                width = HORIZ_BLOCK_SIZE
                height = VERT_BLOCK_SIZE

                block = bb_map[r:r+VERT_BLOCK_SIZE, c:c+HORIZ_BLOCK_SIZE]
                score = torch.sum(block).item()
                # If more than have the pixels are 1, classify as bbox car
                if score > (BLOCK_AREA) * score_threshold:
                    top_left, width, height = go('right', top_left, width, height, bb_map)
                    top_left, width, height = go('left', top_left, width, height, bb_map)
                    top_left, width, height = go('up', top_left, width, height, bb_map)
                    top_left, width, height = go('down', top_left, width, height, bb_map)
                    top_left, width, height = go('left', top_left, width, height, bb_map)
                    top_left, width, height = go('right', top_left, width, height, bb_map)

                    bboxes.append((top_left, width, height))
                    bb_map[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width] = 0  


        new_bboxes = []
        for x in bboxes:
            _, width, height = x
            if width * height > 300:
                if width > 150:
                    car_width = 55
                    dist = 0
                    while dist < width:
                        new_bboxes.append(((x[0][0], x[0][1] + dist), car_width, height))
                        dist += car_width
                elif width > 100:
                    new_bboxes.append((x[0], width * 0.5, height))
                    new_bboxes.append(((x[0][0], x[0][1] + (width * 0.5)), width * 0.5, height))
                else:
                    new_bboxes.append(x)

        bboxes = new_bboxes

        return bboxes
 
    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        road_maps = []
        self.model.eval()
        y_hat, y_count, segmentation = self.model(samples)
        for seg in segmentation:
            road_map = self.faster_reconstruct_from_bins(seg, 5, 0.4)
            road_maps.append(road_map)
        
        road_maps = torch.stack(road_maps).cuda()
        return road_maps
