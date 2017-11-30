#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
#from matplotlib import pypltsplt

# some change
#kmnfwkonf

cv2.namedWindow('Depth')


def main():
    keep_running = True


    while keep_running:
        depth_frame = np.array(get_depth())
        color_frame = np.array(get_video())
        #only works to about 90
        closest = find_min_idx(depth_frame)
        radius = 50
        cv2.circle(depth_frame, closest, radius, (0,255,0),4)
        cv2.imshow('Depth', depth_frame)
        if cv2.waitKey(10) == 27:
            keep_running = False

        print closest





def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k%ncol, k/ncol


if __name__ == "__main__":
    main()