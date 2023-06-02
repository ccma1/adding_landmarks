""" Produces a Gaze Data file

Given Alvaro's mediapipe file and gaze file, it will produce a new gaze file.
The gaze file will have an indicator for if the gaze is on the person's face, 
body, or eye.

It will also produce a new file with indicators for if both people are looking
each other's face, body, and are making eye contact. 
"""
import os
import pandas as pd

from add_landmarks_final import AddLandmarkInfo

def GetGazeOnFaceData(mediapipe_file: str, name_to_save: str):
    """
    Expect around ~40 FPS for processing time.
    """
    g1_gaze_df = AddLandmarkInfo(mediapipe_file)
    g1_gaze_df.to_csv(name_to_save, index=False, na_rep='nan')

def ProcessAll(json_dir: str):
    for fname in os.listdir(json_dir):
        if fname == "20Mar_S1_g_Data.json":
            continue
        print(f'Adding Facemesh Data for {fname}')
        GetGazeOnFaceData(json_dir + fname, fname[:-9]+"face_data.csv")
        print(f'Finished, saved to {fname[:-9]+"face_data.csv"}')

if __name__ == "__main__":
    # g1_mediapipe_file = "/home/Pupilproject/PupilProject/OutputFiles/20Mar_S1_g_Data.json"
    # g1_face_file = "./results/20Mar_S1_g1_Face_Data.csv"

    # print(f"Adding Facemesh Data for {g1_face_file}")
    # GetGazeOnFaceData(g1_mediapipe_file, g1_face_file)
    # print("Finished")
    ProcessAll("/home/Pupilproject/PupilProject/OutputFiles/")