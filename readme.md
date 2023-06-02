
Usage:
 - gaze_on_landmarks.py can process one or all jsons in a given directory. The directory that contains jsons is /home/Pupilproject/PupilProject/OutputFiles/
 - add_landmarks_final.py processes facemeshes. There is a paramater at the top of the file to define the threshold from which a point is considered on the face. This threshold is only used for the face and not other facial features because they're spaced close together and I've defined the facial features to be bigger than what it should be according to the mesh such that it should be valid.
 - landmark_test_3.py provides unit tests for each function. I've just used a facemesh chosen from one of our videos. The tests should also plot the points so you can see what's happening.

Notes:
1. We build a convex hull for the face because the face silhouette isn't defined precisely. 

There are points inside the face mesh that stick out past the face silhouette as defined on the mediapipe github here: 
https://github.com/google/mediapipe/issues/4435
See "invalid_silhouette.png" for an example. The silhouette of the face doesn't contain all the points. See bottom right of face.

2. We need to consider all the points when finding the closest points.
It isn't sufficient to only consider the points of the closest feature or points that aren't contained by a feature.
Consider the following layout of points where * denotes a mediapipe point and G denotes the gaze point. Observe that G is closer to the outside points.
                 *
                 |
(inside feature) G * (outside feature)
                 |
                 *

3. Here are clarifications for mediapipe annotations:
    - https://github.com/google/mediapipe/blob/7c28c5d58ffbcb72043cbe8c9cc32b40aaebac41/mediapipe/python/solutions/face_mesh_connections.py
        - RIGHT is OUR LEFT looking at the screen
        - LEFT is OUR RIGHT
        - LEFT IRIS is OUR LEFT
        - RIGHT IRIS is OUR RIGHT
    - https://github.com/google/mediapipe/blob/7c28c5d58ffbcb72043cbe8c9cc32b40aaebac41/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png 
        - Visualization of the points is here. Note: IRIS points are not visible

4. The face feature polygons are defined according to facial_features.png
