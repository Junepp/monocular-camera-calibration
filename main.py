import os
import cv2
import yaml
import datetime
from typing import Dict


def set_config_with_yaml() -> Dict:
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def save_frames(config: Dict) -> str:
    font = cv2.FONT_HERSHEY_COMPLEX

    # create frames directory
    str_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_path = os.path.join('output', str_datetime)

    os.makedirs(os.path.join(root_path, 'frames'), exist_ok=False)

    # get settings
    camera_device_id = config['camera_idx']
    width = config['camera_width']
    height = config['camera_height']
    number_to_save = config['capture']
    cooldown_time = config['interval']

    # open video stream and change resolution.
    # Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv2.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)

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

        if not start:
            cv2.putText(frame_with_text, "Press SPACEBAR to start collection frames", (50, 50), font, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv2.putText(frame_with_text, f'Cooldown: {cooldown}', (50, 50), font, 1, (0, 255, 0), 1)
            cv2.putText(frame_with_text, f'Num frames: {saved_count}', (50, 100), font, 1, (0, 255, 0), 1)

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join(root_path, 'frames', f'frame{saved_count}.png')
                cv2.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv2.imshow('frame with text', frame_with_text)
        k = cv2.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv2.destroyAllWindows()

    return root_path


if __name__ == '__main__':
    config = set_config_with_yaml()
    saved_frame_path = save_frames(config)
