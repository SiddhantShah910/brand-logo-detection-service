import cv2
import yt_dlp
import os


url = "https://www.youtube.com/watch?v=P1siFAXZeaU"
output_file = "../data/input_video.mp4"
ydl_opts = {
    'outtmpl': output_file,
    'merge_output_format': 'mp4',
}


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

os.makedirs('dataset/images', exist_ok=True)
cam = cv2.VideoCapture(output_file)
count = 0
while True:
    ret, frame = cam.read()
    if not ret: break
    if count % 30 == 0:
        cv2.imwrite(f"dataset/images/frame_{count}.jpg", frame)
    count += 1
cam.release()
