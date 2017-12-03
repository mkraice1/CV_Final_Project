import numpy as np
from numpy import linalg as LA

"""
this program provides:

function to map 3D positions of kinect skeleton points to joint rotation angles, which can be send to baxter.
function to generate testing data to test the first function

skeleton points (num=8): right_shoulder, left_shoulder, left_hip, elbow, wrist, hand, thumb, handtip
each points should be the the form of 1 by 2 numpy array

joint rotations (num=7): shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, hand_yaw, wrist_yaw
each rotation value is a scalar, in radians

Implementation is based on paper "Teleoperation Control of Baxter Robot based on Human Motion Capture"
"""

def points_rotation_compute(left_hip, right_shoulder, left_shoulder, left_elbow, left_wrist, left_hand, left_thumb, left_handtip):
    # 1 solve for shoulder_pitch
    vec_EO = np.subtract(left_hip - left_shoulder)
    vec_EA = np.subtract(left_elbow - left_shoulder)
    shoulder_pitch = np.arccos(np.dot(vec_EO, vec_EA)/(LA.norm(vec_EO,2)*LA.norm(vec_EA,2)))
    # 2 solve for elbow_pitch
    vec_AB = np.subtract(left_wrist - left_elbow)
    elbow_pitch = np.arccos(np.dot(vec_EA, vec_AB)/(LA.norm(vec_EA, 2)*LA.norm(vec_AB, 2)))
    # 3 solve for shoulder_yaw
    vec_ED = np.subtract(right_shoulder - left_shoulder)
    norm_OED = np.cross(vec_ED, vec_EO)
    norm_OEA = np.cross(vec_EO, vec_EA)
    shoulder_yaw = np.arccos(np.dot(norm_OEA, norm_OED)/(LA.norm(norm_OEA, 2)*LA.norm(norm_OED, 2)))
    # 4 solve for elbow_roll
    vec_EM = np.cross(vec_ED, vec_EA)
    vec_AH = np.cross(vec_EA, vec_AB)
    vec_BI = np.subtract(left_thumb - left_wrist)
    vec_BN = np.cross(vec_AB, vec_BI)
    elbow_roll = np.arccos(np.dot(vec_BN, vec_AH)/(LA.norm(vec_BN, 2) * LA.norm(vec_AH, 2)))
    # 5 solve for shoulder roll
    shoulder_roll = np.arccos(np.dot(vec_AH, vec_EM) / (LA.norm(vec_AH, 2) * LA.norm(vec_EM, 2)))
    # 6 solve for hand yaw
    vec_KI = np.subtract(left_thumb, left_hand)
    vec_KC = np.subtract(left_handtip - left_hand)
    norm_IKC = np.cross(vec_KI, vec_KC)
    hand_yaw = np.pi/2 - np.arccos(np.dot(vec_AB, norm_IKC) / (LA.norm(vec_AB, 2) * LA.norm(norm_IKC, 2)))
    # 7 solve for wrist yaw
    vec_BK = np.subtract(left_hand - left_wrist)
    vec_x7 = vec_BK/np.norm(vec_BK)
    vec_z7 = np.cross(vec_BK, vec_BI)/np.norm(np.cross(vec_BK, vec_BI))
    vec_y7 = np.cross(vec_z7, vec_x7)
    # assume k1=1
    k2 = (-(vec_BI[0]+vec_BK[0]))/(vec_BK[0]*vec_AB[0]+vec_BK[1]*vec_AB[1])
    vec_x5 = (vec_BI + k2*vec_BK)/np.norm(vec_BI + k2*vec_BK)
    wrist_yaw = np.cross(vec_y7, vec_x5)

    return shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, hand_yaw, wrist_yaw


def skeleton_points_generation(shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, hand_taw, wrist_yaw):
    # define parameters for a virtual arm
    # units are in cm
    shoulders_dist = 30
    shoulder_elbow_dist = 20
    elbow_wrist_dist = 25
    wrist_hand_dist = 5
    wrist_thumb_dist = 15
    hand_handtip_dist = 15
    shoulder_hip = 45

    # generate 3D points

    return right_shoulder, left_shoulder, left_hip, elbow, wrist, hand, thumb, handtip


def main():
    # requires 8 inputs from skeleton detection
    ################
    left_hip =
    right_shoulder =
    left_shoulder =
    left_elbow =
    left_wrist =
    left_hand =
    left_thumb =
    left_handtip =
    ################
    shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, hand_yaw, wrist_yaw = points_rotation_compute(left_hip, right_shoulder, left_shoulder, left_elbow, left_wrist, left_hand, left_thumb, left_handtip)
    output = np.array(shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_roll, hand_yaw, wrist_yaw)
    print output

if __name__ == "__main__":
    main()