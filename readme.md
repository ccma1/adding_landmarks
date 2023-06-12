See add_landmarks_final.py for key of mapping

Usage:
 - gaze_on_landmarks.py can process one or all jsons in a given directory. The directory that contains jsons is /home/Pupilproject/PupilProject/OutputFiles/
 - add_landmarks_final.py processes facemeshes. 
 - landmark_test_final.py provides unit tests for each function. I've just used a facemesh chosen from one of our videos. The tests should also plot the points so you can see what's happening.


The face feature polygons are defined according to facial_features.png
Features are highlighted in blue, except for mouth which is highlighted in green. 

![image](https://github.com/ccma1/adding_landmarks/assets/79416075/9e2c6973-08e3-4694-9b5d-408665e77954)

On a facemesh it looks like:
![image](https://github.com/ccma1/adding_landmarks/assets/79416075/f827d6dc-52d8-4061-99eb-b606252a84f8)

Notes:
1. We build a convex hull for the face because the face silhouette isn't defined precisely. 

There are points inside the face mesh that stick out past the face silhouette as defined on the mediapipe github here: 
https://github.com/google/mediapipe/issues/4435
See below for an example. The silhouette of the face doesn't contain all the points. See bottom right of face.

![image](https://github.com/ccma1/adding_landmarks/assets/79416075/a47093cf-a451-4434-8354-874743394e21)

2. Because the convex hull is too conservative when we want to identify if the gaze point is on the face, we can relax the constraint for if the gaze is on the face or not. 
This is because the facemesh is too strict and doesn't include hair. We also need to account for the fact that our calibration may be off. 
We relax it as follows:
 * Create the convex hull of the face
 * Get the width and height of the convex hull from Shapely's minimum rotated rectangle https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html#shapely.Polygon.minimum_rotated_rectangle
 * Take the max over the width and the height, and define this as W
 * Define a buffer that is 20% or 30% or W, and any point within the convex hull or the convex hull expanded by 0.2W or 0.3W will be defined as inside the face or not respectively. For example, if gaze_on_20_buffer is 1, this means that: distance from the gaze point - 0.2 * W <= 0

A visualization of this is below:
![image](https://github.com/ccma1/adding_landmarks/assets/79416075/bfaf4382-8629-4a4e-8339-f4883975c51c)

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
