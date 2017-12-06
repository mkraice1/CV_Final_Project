#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
from body_coord import body_coord, position_3D, plot_vector
# some change
# test:ethan2

cv2.namedWindow('Depth')


def main():
    keep_running = True


    while keep_running:
        depth_frame = np.array(get_depth())
        color_frame = np.array(get_video())
        #only works to about 90

        hand_position = find_min_idx(depth_frame)
        #print hand_position
        hand_depth = depth_frame[hand_position[0]][hand_position[1]]
        #print hand_depth

        # segment the hand out
        hand_depth_thresh = 20
        hand_area = np.where(np.logical_and(depth_frame <= (hand_depth + hand_depth_thresh), depth_frame >= hand_depth_thresh))

        # detemine hand state (open = 1, close = 0), based on a threshold value
        hand_area_thresh = 250000
        hand_area = hand_area[0]
        hand_area = hand_area.size
        print hand_area

        hand_state = 0
        if hand_area >= hand_area_thresh:
            hand_state = 1

        body_frame = body_coord(depth_frame)
        hand_position_3D = position_3D(hand_position, hand_depth, body_frame)
        output_message = merge_info(hand_position_3D, hand_state)
        # make this a ros topic for baxter to receive, scale z (depth) into a suitable value, current scale is 500
        print output_message

        color_frame_vector = plot_vector(color_frame, hand_position, body_frame)
        cv2.imshow('body frame', color_frame_vector)

        if cv2.waitKey(10) == 27:
            keep_running = False


def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])

def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return (k%ncol, k/ncol)

def merge_info(hand_position_3D, hand_state):
    output_message = np.array([hand_position_3D[0],hand_position_3D[1],hand_position_3D[2],hand_state])
    return output_message

if __name__ == "__main__":
    main()