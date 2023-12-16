import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
import time
import pickle
from utils.plots import output_to_keypoint, colors, plot_one_box_kpt

# Configurations
view_img = False
save_demo_video = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

filepath = "data_new/EhtT5IJRSTE_(313.82-314.65-315.6)_train.mp4"
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

cap = cv2.VideoCapture(filepath)

if not cap.isOpened():
    raise TypeError("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

total_fps = 0
frame_count = 0
time_list = []  # list to store time
fps_list = []  # list to store fps

if save_demo_video:
    out_filename = f"{'.'.join(filepath.split('.')[:-1])}_annotated.{filepath.split('.')[-1]}"
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    print(out_filename, frame_width, frame_height)

outputList = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = frame.resize((640, 360))
    orig_image = frame  # store frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, frame_width, stride=64, auto=True)[0]
    image = cv2.resize(image, (1920, 1088))
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    # image = image.to(device)  #convert image data to device
    # image = image.float() #convert image to float precision (cpu)
    if torch.cuda.is_available():
        image = image.half().to(device)
    start_time = time.time()  # start time for fps calculation

    with torch.no_grad():  # get predictions
        # print(image)
        print('image.shape:', image.shape)
        output_data, _ = model(image)  # problem

    output_data = non_max_suppression_kpt(output_data,  # Apply non-max suppression
                                          0.25,  # Conf. Threshold.
                                          0.65,  # IoU Threshold.
                                          nc=model.yaml['nc'],  # Number of classes.
                                          nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                          kpt_label=True)

    output = output_to_keypoint(output_data)
    outputList.append(output)

    if view_img or save_demo_video:
        im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
        im0 = im0.cpu().numpy().astype(np.uint8)

        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        for i, pose in enumerate(output_data):  # detections per image
            if len(output_data):  # check if no pose
                for c in pose[:, 5].unique():  # Print results
                    n = (pose[:, 5] == c).sum()  # detections per class
                    print("No of Objects in Current Frame : {}".format(n))

                for det_index, (*xyxy, conf, cls) in enumerate(
                        reversed(pose[:, :6])):  # loop over poses for drawing on frame
                    c = int(cls)  # integer class
                    kpts = pose[det_index, 6:]
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                     line_thickness=3, kpt_label=True, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2])

    # Stream results
    if view_img:
        cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
        cv2.waitKey(1)  # 1 millisecond

    if save_demo_video:
        out.write(im0)  # writing the video frame

    end_time = time.time()  # Calculation for FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    fps_list.append(total_fps)  # append FPS in list
    time_list.append(end_time - start_time)  # append time in list

# pose.close()
cap.release()
out.release()

human_count = max({output.shape[0] for output in outputList})

keypoints = [[] for human_index in range(human_count)]
scores = [[] for human_index in range(human_count)]

data_shape = (...,)
for output in outputList:
    if output.shape[0]:
        data_shape = output[0].shape
        break
zero_data_shape = np.zeros(data_shape)

for frame_index in range(frame_count):
    for human_index in range(human_count):
        data = outputList[frame_index][human_index] \
            if len(outputList[frame_index]) < human_index \
            else zero_data_shape
        data = data[7:].T
        keypoints[human_index].append(np.array([data[::3], data[1::3]]).T)
        scores[human_index].append(data[2::3])

keypoints, scores = np.array(keypoints), np.array(scores)

with open("result.pickle", "wb") as f:
    pickle.dump((keypoints, scores), f, protocol=pickle.HIGHEST_PROTOCOL)
