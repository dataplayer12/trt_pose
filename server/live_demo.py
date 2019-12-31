#!/usr/bin/env python
# coding: utf-8
import torch
from torch2trt import TRTModule
import trt_pose.models
import json
import trt_pose.coco
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
import pdb
import os
import numpy as np
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

#model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

WIDTH, HEIGHT = 224, 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = image[...,::-1] #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

camera = USBCamera(capture_device=1, width=WIDTH, height=HEIGHT, capture_fps=30)
#camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)
outfile='video{}.mp4'.format(len([f for f in os.listdir('./') if f.endswith('.mp4')])+1)
fps=12.0
#sink=cv2.VideoWriter(outfile, 0x31637661, fps, (224,224))
cv2.namedWindow('pose',cv2.WINDOW_NORMAL)
count=1
a=time.time()

with open('keep_recording.txt','w') as f:
    f.write('delete this file to stop recording')

indices={idx:n for idx,n in enumerate(human_pose['keypoints'])}

# pdb.set_trace()

while True:
    if count%100==0:
        print('FPS={:.2f}'.format(100.0/(time.time()-a)))
        a=time.time()
        if not os.path.exists('keep_recording.txt'):
            break

    image=camera.read()
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    sliced=peaks[0,:,0,:]
    
    for idx,row in enumerate(sliced.numpy()):
        point=(224*row).astype(np.int)
        image=cv2.putText(image,indices[idx],(point[1],point[0]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))
    #sink.write(image)
    #image=image[:,::-1,:]
    cv2.imshow('pose',image)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    count+=1

#sink.release()
camera.unobserve_all()