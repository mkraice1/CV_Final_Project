from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

color_map = np.array((
    (255, 106, 0),
    (255, 0, 0),
    (255, 178, 127),
    (255, 127, 127),
    (182, 255, 0),
    (218, 255, 127),
    (255, 216, 0),
    (255, 233, 127),
    (0, 148, 255),
    (72, 0, 255),
    (48, 48, 48),
    (76, 255, 0),
    (0, 255, 33),
    (0, 255, 255),
    (0, 255, 144),
    #(178, 0, 255),
    #(127, 116, 63),
    (127, 63, 63),
    (127, 201, 255),
    (127, 255, 255),
    (165, 255, 127),
    (127, 255, 197),
    (214, 127, 255),
    (161, 127, 255),
    (107, 63, 127),
    (63, 73, 127),
    (63, 127, 127),
    (109, 127, 63),
    (255, 127, 237),
    (127, 63, 118),
    (0, 74, 127),
    (255, 0, 110),
    (0, 127, 70),
    (127, 0, 0),
    (33, 0, 127),
    (127, 0, 55),
    (38, 127, 0),
    (127, 51, 0),
    (64, 64, 64),
    (73, 73, 73),
    (0, 0, 0),
    (191, 168, 247),
    (192, 192, 192),
    (127, 63, 63),
    (127, 116, 63)
))


def color_to_classes(img):
    #lenx, leny = img.size
    n_classes = len(color_map)
    img = img.resize((250,250))
    img_np = np.array(img)

    # Get the alpha channel
    alpha = (img_np[:, :, 3] == 255)

    # make alpha channel into a boolean array
    # alpha = np.equal(alpha, np.ones((leny, lenx))*255)

    img_np = np.array(img.convert('RGB'))
    R = img_np[:, :, 0].astype(int)
    G = img_np[:, :, 1].astype(int)
    B = img_np[:, :, 2].astype(int)
    class_frame = np.zeros((250, 250))
    for i in range(n_classes):
        result = np.logical_and(alpha, np.logical_and(
            np.logical_and((abs(R - color_map[i, 0]) < 3), (abs(G - color_map[i, 1]) < 3)),
            (abs(B - color_map[i, 2]) < 3)))
        
	class_frame[result] = i+1

    #class_frame = cv2.resize(class_frame.astype(np.uint8),(250,250))
    #class_frame = Image.fromarray(class_frame)

    #class_frame = np.swapaxes(class_frame,0,2)
    # class_frame = Image.fromarray(class_frame.astype('uint8'), 'RGB')
    #return np.swapaxes(class_frame,1,2)
    return class_frame

##########could be changed#############
#img = Image.open('./easy-pose/train/1/images/groundtruth/Cam1/mayaProject.000002.png')
#output = color_to_classes(img)
#print np.unique(output)
#output = np.resize(output[[4]], (250,250))
#plt.imshow(output)
#print np.unique(output)
#plt.show()
