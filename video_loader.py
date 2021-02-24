import cv2

def getFirstFrame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    vidcap.retrieve()
    if success:
        return image[200:,200:]