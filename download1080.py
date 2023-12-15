import youtube_dl
from moviepy.editor import VideoFileClip
import os

for filename in os.listdir('rne'):
    if 'train' in filename:
        continue
    id = filename.split('(')[0][:-1]
    print('id1:', id)
    if id[0] == '.':
        id = id[2:]
    print('id:', id)
    interval = list(map(float, filename.split('(')[1].split(')')[0].split('-')))
    ydl_opts = {
        'format': "bestvideo[height<=1080][ext=mp4][vcodec!*=av01]",
        'outtmpl': f"./data_original/%(id)s.%(ext)s"
    }
    url = f"https://youtube.com/watch?v={id}"

    if not os.path.exists(f"./data_original/{id}.mp4"):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', None)
        print(f'VIDEO_TITLE={video_title}')
    else:
        print('video already exists: skip')

    videoFileClip = VideoFileClip(f"data_original/{id}.mp4")

    targetPath1 = f"data_new/{id}_({interval[0]}-{interval[1]}-{interval[2]}).mp4"
    videoFileClip.subclip(interval[0], interval[1]).write_videofile(targetPath1, fps=24)
    targetPath2 = f"data_new/{id}_({interval[0]}-{interval[1]}-{interval[2]})_train.mp4"
    videoFileClip.subclip(interval[1], interval[2]).write_videofile(targetPath2, fps=24)

    videoFileClip.reader.close()
    
