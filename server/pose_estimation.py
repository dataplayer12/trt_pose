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
import matplotlib.pyplot as plt
import blynklib

threshold=45
token='2w36TnckERRPKCxIzezteRbT-u0vg91n'
blynk = blynklib.Blynk(token)

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

WIDTH, HEIGHT = 224, 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    image = image[...,::-1]
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

camera = USBCamera(capture_device=1, width=WIDTH, height=HEIGHT, capture_fps=30)
cv2.namedWindow('pose',cv2.WINDOW_NORMAL)
framecount=1
indices={idx:n for idx,n in enumerate(human_pose['keypoints'])}
positions={n:[[0,0]]*5 for n in human_pose['keypoints']}
detected=lambda n: positions[indices[n]][-1]!=[0,0]
groups={'head':[0,1,2,3,4],'mid':[5,6,17],'bottom':[11,12]}
group_avgs={'head':[],'mid':[],'bottom':[]}
group_dets={'head':False,'mid':False,'bottom':False}
angles=[]
spine_angle=0

@blynk.handle_event('read V11')
def write_virtual_pin_handlerv11(pin):
    #print(WRITE_EVENT_PRINT_MSG.format(pin, value))
    global spine_angle
    blynk.virtual_write(11, spine_angle)
    if spine_angle>threshold:
        blynk.notify('Correct your posture')

@blynk.handle_event('read V2')
def write_virtual_pin_handlerv2(pin):
    #print(WRITE_EVENT_PRINT_MSG.format(pin, value))
    global spine_angle
    blynk.virtual_write(2, (spine_angle!=0))

a=time.time()
while True:
    blynk.run()
    if framecount%100==0:
        print('FPS={:.2f}'.format(100.0/(time.time()-a)))
        a=time.time()
    image=camera.read()
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    #draw_objects(image, counts, objects, peaks)
    try:
        sliced=(224*(peaks[0,:,0,:].numpy())).astype(np.int)
        for idx,point in enumerate(sliced):
            positions[indices[idx]].append(list(point))
            positions[indices[idx]].pop(0)
            #image=cv2.putText(image,indices[idx],(point[1],point[0]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))
        for n,l in groups.items():
            for p in l:
                if detected(p):
                    group_avgs[n].append(sliced[p])
            if len(group_avgs[n])>0:
                group_avgs[n]=np.mean(group_avgs[n],axis=0).astype(np.int)
                group_dets[n]=True
                image=cv2.putText(image,n,(group_avgs[n][1],group_avgs[n][0]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))
            else:
                group_avgs[n]=[0,0]
        
        topgroup='head' if group_dets['head'] else 'mid'
        bottomgroup='mid' if group_dets['mid'] else 'bottom'
        if topgroup==bottomgroup:
            continue

        star=(group_avgs[topgroup]-group_avgs[bottomgroup])
        spine_angle=0.8*spine_angle+180*0.2*np.arctan(star[1]/float(star[0]))/np.pi
        #print(spine_angle)
        angles.append(spine_angle)
    except Exception as e:
        #print(str(e))
        spine_angle=0.0
    group_avgs={'head':[],'mid':[],'bottom':[]}
    group_dets={'head':False,'mid':False,'bottom':False}
    cv2.imshow('pose',image)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    framecount+=1

camera.unobserve_all()
cv2.destroyAllWindows()
plt.plot(np.arange(len(angles)),angles)
plt.show()
