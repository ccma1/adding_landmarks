""" Produces a Gaze Data file

Given Alvaro's mediapipe file and gaze file, it will produce a new gaze file.
The gaze file will have an indicator for if the gaze is on the person's face, 
body, or eye.

It will also produce a new file with indicators for if both people are looking
each other's face, body, and are making eye contact. 

You can also visualize the mediapipe face and body points if desired.

The data regarding eye points and location of each point is here:
 - Image: https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 - https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt
 - https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

Note: the 2nd and 3rd give the same information, but left and right eyes are flipped. 
 - In the 2nd file, Left eye := Left for the viewer's perspective ie. us.
 - In the 3rd file, Left eye := Left of the person's whose face is mapped ie. our right

WE USE THE THIRD FILD as reference because it provides more fine grained information such as nose location.
"""

from add_landmarks import AddLandmarkInfo

if __name__ == "__main__":
    g1_mediapipe_file = "./data/march22/session1/Result_22Mar_S1_g1_DataMediapipe.csv"
    g2_mediapipe_file = "./data/march22/session1/Result_22Mar_S1_g2_DataMediapipe.csv"
    g1_gaze_file      = "./data/march22/session1/22Mar_S1_generalTimestamp_g1.csv"
    g2_gaze_file      = "./data/march22/session1/22Mar_S1_generalTimestamp_g2.csv"
    g1_gaze_out_name = './results/march22_session1/22Mar_S1_generalTimestamp_g1_with_landmarks.csv'
    g2_gaze_out_name = './results/march22_session1/22Mar_S1_generalTimestamp_g2_with_landmarks.csv'
    out_file_name     = "./results/march22_session1/session1_timestamp_gaze_face_indicator.csv"

    g1_gaze_df = AddLandmarkInfo(g1_gaze_file, g1_mediapipe_file)
    # g2_gaze_df = AddLandmarkInfo(g2_gaze_file, g2_mediapipe_file)
    # AddGazeFaceBodyInfo(g1_gaze_file, g1_mediapipe_file, g1_gaze_out_name)
    # AddGazeFaceBodyInfo(g2_gaze_file, g2_mediapipe_file, g2_gaze_out_name)
    # CombineGazeData(g1_gaze_out_name, g2_gaze_out_name, out_file_name)
    # ShowLookingAt(out_file_name)