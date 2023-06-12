""" Produces a Gaze Data file

Given Alvaro's mediapipe file and gaze file, it will produce a new gaze file.
The gaze file will have an indicator for if the gaze is on the person's face, 
body, or eye.

You can also visualize the mediapipe face and body points if desired using the test files.

Read the Readme for how the points are determined and the codes for the points.
"""
import os
import pandas as pd

from add_landmarks_final import AddLandmarkInfo

def GetGazeOnFaceData(mediapipe_file: str, name_to_save: str):
    """
    Expect around ~80 FPS for processing speed.

    373.96822142601013 seconds for 30596 frames
    """
    g1_gaze_df = AddLandmarkInfo(mediapipe_file)
    g1_gaze_df.to_csv(name_to_save, index=False, na_rep='nan')

def ProcessAll(json_dir: str, save_to_dir: str):
    for fname in os.listdir(json_dir):
        if ((fname[:-9]+"face_data.csv") in os.listdir(save_to_dir)):
            print(f"Skipping {fname}: already processed")
            continue
        if (os.stat(json_dir + fname).st_size == 0):
            print(f'Skipping {fname}: file size 0')
            continue

        print(f'Adding Facemesh Data for {fname}')
        GetGazeOnFaceData(json_dir + fname, fname[:-9]+"face_data.csv")
        print(f'Finished, saved to {fname[:-9]+"face_data.csv"}')

if __name__ == "__main__":
    ProcessAll(
        json_dir="/home/Pupilproject/PupilProject/OutputFiles_new/",
        save_to_dir="./facemesh_results_new/"
    )