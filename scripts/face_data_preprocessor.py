import os
import numpy as np
import cv2
import dlib
import glob
import argparse
from skimage import io
from pathlib import Path


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_frames_and_keypoints_for_one_video(video_idx, video_path):
    videos_dir = str(Path(video_path).parent)
    frames_dir = videos_dir.replace("/original", "") + "_img"
    os.makedirs(os.path.join(frames_dir, str(video_idx).zfill(4)), exist_ok=True)
    keypoints_dir = videos_dir.replace("/original", "") + "_keypoints"
    os.makedirs(os.path.join(keypoints_dir, str(video_idx).zfill(4)), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    read_frame_idx = 0
    save_frame_idx = 0
    while True:
        success, image = cap.read()

        # end of video
        if not success:
            break

        dets = detector(image, 1)

        # detected at least one face
        if len(dets) > 0:

            # Use every 6th frame where we detected a face
            if (read_frame_idx + 1) % 6 != 0:
                read_frame_idx += 1
                continue

            cv2.imwrite(
                os.path.join(
                    frames_dir, str(video_idx).zfill(4), str(save_frame_idx).zfill(4)
                )
                + ".jpg",
                image,
            )

            shape = predictor(image, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b, 0] = shape.part(b).x
                points[b, 1] = shape.part(b).y
            keypoint_path = (
                os.path.join(
                    keypoints_dir, str(video_idx).zfill(4), str(save_frame_idx).zfill(4)
                )
                + ".txt"
            )
            np.savetxt(keypoint_path, points, fmt="%d", delimiter=",")
            read_frame_idx += 1
            save_frame_idx += 1


def extract_frames_and_keypoints_for_all_videos(videos_dir):
    for split in ["train", "val", "test"]:
        current_videos_dir = os.path.join(videos_dir, split, "original")
        video_paths = [
            os.path.join(videos_dir, split, "original", video_name)
            for video_name in sorted(os.listdir(current_videos_dir))
            if len(video_name) >= 4 and video_name[-4:] == ".avi"
        ]

        for video_idx, video_path in enumerate(video_paths):
            extract_frames_and_keypoints_for_one_video(video_idx, video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preprocessing script")

    parser.add_argument(
        "--videos_dir", type=str, help="Directory where the videos are located."
    )

    args = parser.parse_args()

    videos_dir = args.videos_dir

    videos_dir = "/cluster/scratch/aarslan/virtual_humans_data/datasets/face"

    extract_frames_and_keypoints_for_all_videos(videos_dir)
