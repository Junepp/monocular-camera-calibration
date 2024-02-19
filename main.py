import os
import cv2
import yaml
import datetime
import numpy as np
from typing import Dict


def set_config_with_yaml() -> Dict:
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def save_frames(config: Dict) -> str:
    font = cv2.FONT_HERSHEY_COMPLEX

    # create save directory
    str_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_path = os.path.join('output', str_datetime)

    os.makedirs(os.path.join(root_path, 'frames'), exist_ok=False)

    # get settings
    camera_device_id = config['camera_idx']
    width = config['camera_width']
    height = config['camera_height']
    number_to_save = config['capture']
    cooldown_time = config['interval']

    cap = cv2.VideoCapture(camera_device_id)

    # set w, h of frame
    w_retval = cap.set(3, width)
    h_retval = cap.set(4, height)

    # # autofocus disable (0 to 1024)
    # af_retval = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_FOCUS, 512)

    print(f'set frame width to {width}? {w_retval}')
    print(f'set frame width to {height}? {h_retval}')
    # print(f'set camera autofoucs disable? {af_retval}')

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_with_text = frame.copy()
        print(frame_with_text.shape)

        if not start:
            cv2.putText(frame_with_text, "Press SPACEBAR to start collection frames", (50, 50), font, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv2.putText(frame_with_text, f'Cooldown: {cooldown}', (50, 50), font, 1, (0, 255, 0), 1)
            cv2.putText(frame_with_text, f'Num frames: {saved_count}', (50, 100), font, 1, (0, 255, 0), 1)

            if cooldown <= 0:
                savename = os.path.join(root_path, 'frames', f'frame{saved_count}.png')
                cv2.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv2.imshow('frame with text', frame_with_text)
        k = cv2.waitKey(1)

        if k == 27: quit()
        if k == 32: start = True
        if saved_count == number_to_save: break

    cv2.destroyAllWindows()

    return root_path


def calibrate_camera_for_intrinsic_parameters(config: Dict, image_root_path: str):
    images_names = os.listdir(f'{image_root_path}/frames')

    # read all frames
    images = [cv2.imread(f'{image_root_path}/frames/{imname}', 1) for imname in images_names]

    # criteria used by checkerboard pattern detector.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = config['checkerboard_rows']
    columns = config['checkerboard_cols']
    world_scaling = config['checkerboard_box_size_scale']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            cv2.imshow('img', frame)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    cv2.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist


# save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, save_path):
    # TODO 1. save params as json type
    # TODO 2. save rmse value
    out_filename = os.path.join(save_path, 'params.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')
    outf.close()


if __name__ == '__main__':
    # LOAD CONFIG
    config = set_config_with_yaml()

    # SAVE FRAMES
    saved_frame_path = save_frames(config=config)

    # CALCULATE INTRINSIC PARAMETERS
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(config=config, image_root_path=saved_frame_path)

    # SAVE INTRINSIC PARAMETERS
    save_camera_intrinsics(cmtx0, dist0, save_path=saved_frame_path)  # this will write cmtx and dist to disk

    print(saved_frame_path)
