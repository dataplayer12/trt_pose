#!/usr/bin/env python
# coding: utf-8
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
from jetcam.usb_camera import USBCamera
import torch
from torch2trt import TRTModule
import trt_pose.models
import json
import trt_pose.coco
import torchvision.transforms as transforms
import PIL.Image
import pdb
import os
import numpy as np
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import matplotlib.pyplot as plt
import blynklib
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)

threshold=45
with open('authtoken.txt','r') as f:
    token=f.read()
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

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
camera=USBCamera(capture_device=0, width=224, height=224, capture_fps=30)
time.sleep(1.0)
indices={idx:n for idx,n in enumerate(human_pose['keypoints'])}
positions={n:[[0,0]]*5 for n in human_pose['keypoints']}
detected=lambda n: positions[indices[n]][-1]!=[0,0]
groups={'head':[0,1,2,3,4],'mid':[5,6,17],'bottom':[11,12]}
group_avgs={'head':[],'mid':[],'bottom':[]}
group_dets={'head':False,'mid':False,'bottom':False}
spine_angle=0

@blynk.handle_event('read V11')
def write_virtual_pin_handlerv11(pin):
    global spine_angle, threshold
    blynk.virtual_write(11, spine_angle)
    blynk.virtual_write(10, threshold)
    if spine_angle>threshold:
        blynk.notify('Correct your posture')

@blynk.handle_event('write V12')
def write_virtual_pin_handler(pin,value):
    global threshold
    threshold=int(value[0])
    blynk.virtual_write(12, threshold)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_pose(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, angles, spine_angle
    framecount=1
    a=time.time()
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        image=camera.read()
        blynk.run()
        if framecount%100==0:
            print('FPS={:.2f}'.format(100.0/(time.time()-a)))
            a=time.time()
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(image, counts, objects, peaks)
        try:
            sliced=(224*(peaks[0,:,0,:].numpy())).astype(np.int)
            for idx,point in enumerate(sliced):
                positions[indices[idx]].append(list(point))
                positions[indices[idx]].pop(0)
                #image=cv2.putText(image,indices[idx],(point[1],point[0]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))
            for n,l in groups.items():
                for p in l:
                    if detected(p):
                        group_avgs[n].append(list(sliced[p]))
                if len(group_avgs[n])>0:
                    group_avgs[n]=np.mean(group_avgs[n],axis=0).astype(np.int)
                    group_dets[n]=True
                    #image=cv2.putText(image,n,(group_avgs[n][1],group_avgs[n][0]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255))
                else:
                    group_avgs[n]=[0,0]
            
            topgroup='head' if group_dets['head'] else 'mid'
            bottomgroup='bottom' if group_dets['bottom'] else 'mid'
            #print(topgroup,bottomgroup)
            star=(group_avgs[topgroup]-group_avgs[bottomgroup])
            spine_angle=abs(0.8*spine_angle+180*0.2*np.arctan(star[1]/float(star[0]))/np.pi)
            if topgroup==bottomgroup:
                spine_angle=0.0
        except Exception as e:
            print(str(e))
            spine_angle=0.0
        group_avgs={'head':[],'mid':[],'bottom':[]}
        group_dets={'head':False,'mid':False,'bottom':False}
        framecount+=1
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(image, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (5, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        with lock:
            outputFrame = image.copy()
        
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str,
        default='0.0.0.0', help="ip address of the device")
    ap.add_argument("-o", "--port", type=int,
        default=8000, help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_pose, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
    camera.unobserve_all()
