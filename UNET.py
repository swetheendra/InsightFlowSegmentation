import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image


class DownSamplingConvBlock(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.conv_layers = []

        for i in range(len(layers)-1):
            layer = nn.Conv2d(in_channels=layers[i], out_channels=layers[i+1], kernel_size=4, stride=2, padding=1)
            self.conv_layers.append(layer)
        
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        intermediate_states = []
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = F.relu(x)

            if i != len(self.conv_layers)-1:
                intermediate_states.append(x.clone())
        
        return x,intermediate_states

class UpSamplingConvBlock(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        self.convs = []
        for i in range(len(layers)-1):
            layer = nn.Conv2d(in_channels=2*layers[i] if i > 0 else layers[i], out_channels=layers[i+1], kernel_size=3, stride=1, padding=1)
            self.convs.append(layer)
        
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, intermediate_states):

        for i in range(len(self.convs)):
            x = self.upsampler(x)
            x = self.convs[i](x)
            if i != len(self.convs)-1:
                x = F.relu(x)
                total_inter_states = len(intermediate_states)
                x = torch.cat([x, intermediate_states[total_inter_states-i-1]], axis=1)
        return x
        

class UNET(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels=1, layers=[16,32,64, 128, 256, 512]):
        super().__init__()

        downsampling_layers = [in_channels]+layers
        upsampling_layers = layers[::-1]+[out_channels]

        self.upsamplingLayers = UpSamplingConvBlock(upsampling_layers)
        self.downsamplingLayers = DownSamplingConvBlock(downsampling_layers)

    def forward(self, x):
        downsampled, intermediate_state = self.downsamplingLayers(x)
        output = self.upsamplingLayers(downsampled, intermediate_state)
        return output



def json_to_mask(height, width, polygons):
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons:
        points = np.array(polygon).reshape(-1,2).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    
    return mask

class FootBallSegmentationDataSet(Dataset):
    def __init__(self, image_dir, annotations_dir, transform):
        self.transform = transform
        self.annotations = defaultdict(list)
        self.image_dir = image_dir
        self.img = {}

        with open(annotationsPath, 'r') as file:
            data = json.load(file)
            annotations_list = data["annotations"]
            images = data["images"]

            for image in images:
                imgId = image['id']
                self.img[imgId] = image

            for annotation in annotations_list:
                imgId = annotation['image_id']
                self.annotations[imgId].append(annotation)
        
        self.img_ids = list(self.img.keys())
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, ind):
        idx = self.img_ids[ind]
        imageIdx = self.img[idx]
        imagepath = os.path.join(self.image_dir, imageIdx['file_name'])
        img = Image.open(imagepath).convert("RGB")

        annts = [annotation['segmentation'][0] for annotation in self.annotations[idx]]

        mask = json_to_mask(imageIdx['height'], imageIdx['width'], annts)
        img_np = np.array(img)
        augmented = self.transform(image=img_np, mask=mask)
        maskDim = augmented['mask'].unsqueeze(0)
        return augmented['image'], maskDim

