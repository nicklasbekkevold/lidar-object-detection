import cv2
import sys
import os

class avi2jpg:
    vid = None

    def __init__(self, video):
        self.video = video
    
    def convert(self, video):
        vidcap = cv2.VideoCapture(video)
        success,image = vidcap.read()
        count = 0
        try:
            os.mkdir("data/"+self.video+"/images")
        except FileExistsError:
            pass
        while success:
            framecount = "{number:06}".format(number=count)
            cv2.imwrite("data/"+self.video+"/images/frame_"+framecount+".jpg", image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

if __name__ == "__main__":
        converter = avi2jpg(sys.argv[1])
        converter.convert(sys.argv[2])