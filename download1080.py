import youtube_dl
from moviepy.editor import VideoFileClip
import os

original_path = '480p_1218'
original_video_path = 'data_original_1218'
output_path = 'data_new_1218'

filenames = os.listdir(original_path)
file_count = len(filenames)

for index, filename in enumerate(filenames):
    if 'train' in filename:
        continue
    id = filename.split('(')[0][:-1]
    if id[0] == '.':
        id = id[2:]
    print(f"index: {index}/{file_count}, id: {id}")
    interval = list(map(float, filename.split('(')[1].split(')')[0].split('-')))
    ydl_opts = {
        'format': "bestvideo[height<=1080][ext=mp4][vcodec!*=av01]",
        'outtmpl': f"./{original_video_path}/%(id)s.%(ext)s"
    }
    url = f"https://youtube.com/watch?v={id}"

    if not os.path.exists(f"./{original_video_path}/{id}.mp4"):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', None)
        print(f'VIDEO_TITLE={video_title}')
    else:
        print('video already exists: skip')

    videoFileClip = VideoFileClip(f"{original_video_path}/{id}.mp4")

    targetPath1 = f"{output_path}/{id}_({interval[0]}-{interval[1]}-{interval[2]}).mp4"
    if os.path.exists(targetPath1):
        print("targetPath1: skip")
    else:
        videoFileClip.subclip(interval[0], interval[1]).write_videofile(targetPath1, fps=24)

    targetPath2 = f"{output_path}/{id}_({interval[0]}-{interval[1]}-{interval[2]})_train.mp4"
    if os.path.exists(targetPath2):
        print("targetPath2: skip")
    else:
        videoFileClip.subclip(interval[1], interval[2]).write_videofile(targetPath2, fps=24)

    videoFileClip.reader.close()
