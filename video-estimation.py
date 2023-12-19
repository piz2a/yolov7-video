import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
import time
import os
import pickle
from utils.plots import output_to_keypoint, colors, plot_one_box_kpt

# Configurations
view_img = False
save_demo_video = False
# select_skeletons = False
coco_center = 1
pickle_path = "../result.pickle"
data_dir = 'data_test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)


def input_int(message):
    try:
        result = int(input(message))
        return result
    except ValueError:
        print("Please enter an integer. Try again.")
        return input_int(message)


def show_skeleton(image, output_data, names):
    im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)

    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
    # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    for i, pose in enumerate(output_data):  # detections per image
        print(output_data)
        if len(output_data):  # check if no pose
            print(f"{i}/{len(output_data)}")
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                print("No of Objects in Current Frame : {}".format(n))

            for det_index, (*xyxy, conf, cls) in enumerate(
                    (pose[:, :6])):  # loop over poses for drawing on frame
                # print(f"- {det_index} / {n}")
                c = int(cls)  # integer class
                kpts = pose[det_index, 6:]
                label = f'{names[c]} No.{det_index}-{conf:.2f}'
                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                 line_thickness=3, kpt_label=True, kpts=kpts, steps=3,
                                 orig_shape=im0.shape[:2])

    return im0


def choose_skeleton_using_keys(log_text):
    key_input = ''
    while True:
        key = cv2.waitKey(0)
        if key in range(ord('0'), ord('9')+1):
            key_input += chr(key)
            print(f"{log_text}{key_input}")
        elif key in [10, 13]:  # Enter
            if key_input:
                break
            print("Please enter a number")
        elif key == 8:  # Backspace
            if key_input:
                key_input = key_input[:-1]
            print(f"{log_text}{key_input}")
        elif key == ord('x'):
            print("Press 'x' again to skip this video.")
            key2 = cv2.waitKey(0)
            if key2 == ord('x'):
                print("Giving up this data")
                return None
            print('Video skipping canceled.')
        else:
            print("Please enter a number")
    return int(key_input)


def video_pose_estimation(data_dir, filename, index, video_count):
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    filepath = f"{data_dir}/{filename}"
    print('filepath:', filepath)
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
    isFirst = True
    image0 = None
    output_data0 = None

    while cap.isOpened():
        # print('test1')
        ret, frame = cap.read()
        if not ret:
            break

        # frame = frame.resize((640, 360))
        orig_image = frame  # store frame
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
        image = letterbox(image, frame_width, stride=64, auto=True)[0]
        image = cv2.resize(image, (1920, 1088))
        # image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        # print('test2')

        # image = image.to(device)  #convert image data to device
        # image = image.float() #convert image to float precision (cpu)
        if torch.cuda.is_available():
            image = image.half().to(device)
        start_time = time.time()  # start time for fps calculation

        # print('test3')
        with torch.no_grad():  # get predictions
            # print(image)
            # print('image.shape:', image.shape)
            print(frame_count, end=' ', flush=True)
            output_data, _ = model(image)  # problem

        # print('test4')
        output_data = non_max_suppression_kpt(output_data,  # Apply non-max suppression
                                            0.25,  # Conf. Threshold.
                                            0.65,  # IoU Threshold.
                                            nc=model.yaml['nc'],  # Number of classes.
                                            nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                            kpt_label=True)
        if isFirst:
            image0 = image
            output_data0 = output_data
            # print('output_data0.shape:', output_data0.shape)
            isFirst = False

        output = output_to_keypoint(output_data)
        # print('output.shape:', output.shape)
        outputList.append(output)

        if view_img or save_demo_video:
            im0 = show_skeleton(image, output_data, names)

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
    if save_demo_video:
        out.release()

    # select the only two skeletons: kicker's and goalkeeper's
    title = f"YOLOv7 Pose Estimation Demo: No. {index} / {video_count}"
    im0 = show_skeleton(image0, output_data0, names)
    im0 = cv2.resize(im0, (960, 540))
    cv2.imshow(title, im0)

    print("Enter Kicker Index")
    key_input = choose_skeleton_using_keys(f"{title} - Kicker: ")
    if key_input is None:
        return None
    kicker_index = int(key_input)
    print("Kicker:", kicker_index)

    print("Enter Goalkeeper Index")
    key_input = choose_skeleton_using_keys(f"{title} - Kicker: {kicker_index} - Goalkeeper: ")
    if key_input is None:
        return None
    goalkeeper_index = int(key_input)
    print("Goalkeeper:", goalkeeper_index)

    """
    im_done = show_skeleton(image0, [output_data0[kicker_index], outp], names)
    im_done = cv2.resize(im_done, (960, 540))
    cv2.imshow(title, im_done)
    """

    human_count = 2  # max({output.shape[0] for output in outputList})

    keypoints = [[] for _ in range(human_count)]
    scores = [[] for _ in range(human_count)]

    data_shape = (...,)
    for output in outputList:
        if output.shape[0]:
            data_shape = output[0].shape
            break
    zero_data = np.zeros(data_shape)[7:].T

    # 이전 프레임의 skeleton과 거리가 가장 가까운 skeleton을 찾고 이 두 skeleton이 같은 사람이라고 가정
    for result_index, needed_index_0 in enumerate([kicker_index, goalkeeper_index]):
        # print("needed_index_0:", needed_index_0)
        prev_min_data = outputList[0][needed_index_0][7:].T
        for frame_index in range(frame_count):
            min_distance_square = frame_width ** 2 + frame_height ** 2 + 1000
            min_data = zero_data
            prev_keypoint = np.array([prev_min_data[::3], prev_min_data[1::3]]).T
            for human_index, data in enumerate(outputList[frame_index]):
                data = data[7:].T
                keypoint = np.array([data[::3], data[1::3]]).T
                distance_square = np.sum(np.power(keypoint[coco_center] - prev_keypoint[coco_center], 2))
                if distance_square < min_distance_square:
                    min_distance_square = distance_square
                    min_data = data
            if min_distance_square > (frame_width ** 2 + frame_height ** 2) / (4 ** 2):
                # 너무 멀 경우 사람 인식을 못하는 것으로 간주하고 이전 포즈를 그대로 유지한다.
                min_data = zero_data
            else:
                # 그렇지 않은 경우에만 이전 포즈가 새 포즈로 바뀐다.
                prev_min_data = min_data
            min_keypoint = np.array([min_data[::3], min_data[1::3]]).T
            keypoints[result_index].append(min_keypoint)
            min_score = min_data[2::3]
            scores[result_index].append(min_score)

    """
    # copying the first frame which was recognized
    for human_index in range(human_count):
        for frame_index in range(frame_count):
            if keypoints[human_index][frame_index].any():  # if the first frame which is not all zeros
                keypoint_first = keypoints[human_index][frame_index]
                score_first = scores[human_index][frame_index]
                for new_frame_index in range(frame_index):
                    keypoints[human_index][new_frame_index] = keypoint_first
                    scores[human_index][new_frame_index] = score_first
                break
    """

    keypoints, scores = np.array(keypoints), np.array(scores)
    return filepath, keypoints, scores


try:
    with open(pickle_path, "rb") as f:
        data_list = pickle.load(f)
except FileNotFoundError:
    data_list = []

list_dir = os.listdir(data_dir)
video_count = len(list_dir)
index = -2
for filename in list_dir:
    cv2.destroyAllWindows()
    index += 2
    if 'train' in filename:
        continue
    result1 = video_pose_estimation(data_dir, filename, index, video_count)
    if result1 is None:
        continue
    result2 = video_pose_estimation(data_dir, filename[:-4] + '_train.mp4', index + 1, video_count)
    if result2 is None:
        continue
    data_list.append([result1, result2])

with open(pickle_path, "wb") as f:
    pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saving Complete")
