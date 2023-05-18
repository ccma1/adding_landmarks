"""Adds Landmark information 

For a given gaze file, creates a new dataframe with the following columns:
 - closest_face_point: Coordinate of closest point from face mesh or None
 - closest_facial_feature: One of {right_eye, left_eye, nose, mouth, none (if no face mesh)}
 - on_facial_region: The facial feature the gaze is inside if face mesh is present and it's inside, defined above,
      or None if no facial mesh or if gaze isn't on any facial feature
 - closest_facial_feature_dist:
    - 0 if inside a facial feature
    - > 0 if gaze point is outside a facial feature
 - on_face: 
    - 1 if gaze is inside face (with margin see below)
    - 0 otherwise
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import csv
import ast
import sympy
import functools

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
from typing import Tuple, List
from sympy import Point, convex_hull, Polygon, N
from operator import itemgetter
from itertools import chain

DEBUG = True

ON_FACE_DISTANCE_THRESHOLD = 0

FEATURE_NAMES = ["right_eye", "left_eye", "mouth", "nose"]

MESH_ANNOTATIONS = {
   # As defined here:
   # https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
  "silhouette": [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ],

  "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
  "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
  "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
  "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
  "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
  "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
  "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

  "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
  "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

  "rightEyeIris": [473, 474, 475, 476, 477],

  "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
  "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
  "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
  "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
  "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
  "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
  "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

  "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
  "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

  "leftEyeIris": [468, 469, 470, 471, 472],

  "midwayBetweenEyes": [168],

  "noseTip": [1],
  "noseBottom": [2],
  "noseRightCorner": [98],
  "noseLeftCorner": [327],

  "rightCheek": [205],
  "leftCheek": [425]
}

MESH_ANNOTATIONS['leftEyebrowUpper'] = MESH_ANNOTATIONS['leftEyebrowUpper'][::-1]
MESH_ANNOTATIONS['rightEyeLower3'] = MESH_ANNOTATIONS['rightEyeLower3'][::-1]
MESH_ANNOTATIONS['lipsUpperOuter'] = MESH_ANNOTATIONS['lipsUpperOuter'][:-1]
MESH_ANNOTATIONS['lipsLowerOuter'] = MESH_ANNOTATIONS['lipsLowerOuter'][::-1]

def GetClosestPoint(gaze_point: sympy.Point, face_points: List[sympy.Point]):
   """
   Returns the closest face point closest to the gaze point
   """
   return functools.reduce(
      lambda p1, p2: p1 if p1.distance(gaze_point) <= p2.distance(gaze_point) else p2,
      face_points
   )

def VisualizeFeatures(gaze_point, feature_list, face_points):
   plt.figure(figsize=(10.88,10.8))
   if gaze_point != None:
      plt.plot(gaze_point[0], gaze_point[1], 'bo')
   x, y = zip(*face_points)
   plt.plot(x, y, 'go',markersize=1)
   for feature_poly in feature_list:
      x, y = zip(*feature_poly.vertices)
      plt.plot(x, y)
      plt.fill(x, y, alpha=0.3)
   plt.axis('equal')
   plt.show()

def GetClosestFeature(gaze_point: sympy.Point, face_points: List[sympy.Point]):
   """
   Finds the face feature or closest face feature closest to the gaze point.

   Returns
   -------
   Str:
      Closest facial feature, one of 
      {right_eye, left_eye, nose, mouth}
   Str:
      Which facial feature gaze is on, if applicable. 
      One of {right_eye, left_eye, nose, mouth, none}
   Float:
      Distance of gaze point from facial feature polygon.
      0 if inside feature else result >= 0
   """
   feature_list = [
      # Note polygon vertices may have fewer than selected because Sympy will remove points lying on line for polygon
      # Right eye from viewer's perspective, clockwise from left corner from viewer's perspective
      Polygon(*(chain(itemgetter(*MESH_ANNOTATIONS['leftEyebrowUpper'])(face_points), 
                      itemgetter(*MESH_ANNOTATIONS['leftEyeLower3'])(face_points)))),
      # Left eye from viewer's perspective, clockwise from left corner from viewer's perspective
      Polygon(*(chain(itemgetter(*MESH_ANNOTATIONS['rightEyebrowUpper'])(face_points),
                      itemgetter(*MESH_ANNOTATIONS['rightEyeLower3'])(face_points)))),
      # Mouth, clockwise from left corner from viewer's perspective
      Polygon(*(chain(itemgetter(*MESH_ANNOTATIONS['lipsUpperOuter'])(face_points),
                      itemgetter(*MESH_ANNOTATIONS['lipsLowerOuter'])(face_points)))),
      # Nose, clockwise from left corner from viewer's perspective
      Polygon(*(chain([itemgetter(*MESH_ANNOTATIONS['noseRightCorner'])(face_points)],
                      [itemgetter(*MESH_ANNOTATIONS['noseTip'])(face_points)],
                      [itemgetter(*MESH_ANNOTATIONS['noseLeftCorner'])(face_points)],
                      [itemgetter(*MESH_ANNOTATIONS['noseBottom'])(face_points)])))
   ]

   min_feature_dist, closest_feature, on_feature = None, None, "none"
   # We check distance because poly.contains doesn't include points on boundary. 0 check bc if inside then non-zero
   for index in range(len(feature_list)):
      feature_poly = feature_list[index]
      gaze_dist = feature_poly.distance(gaze_point)
      gaze_on_feature = feature_poly.encloses_point(gaze_point) or gaze_dist <= ON_FACE_DISTANCE_THRESHOLD
      curr_feature_name = FEATURE_NAMES[index]
      if gaze_on_feature:
         if on_feature != "none":
            print("BAD THRESHOLD: GAZE ON MULTIPLE FACIAL FEATURES")
         min_feature_dist = 0
         closest_feature  = curr_feature_name
         on_feature       = curr_feature_name
      elif min_feature_dist == None or gaze_dist < min_feature_dist:
         min_feature_dist = gaze_dist
         closest_feature = curr_feature_name
   
   if DEBUG:
      if on_feature != "None":
         VisualizeFeatures(gaze_point, feature_list, face_points)
   return closest_feature, on_feature, N(min_feature_dist)

def GazeOnFace(gaze_point: sympy.Point, face_points: List[sympy.Point]): 
   """
   Returns if either:
   1. The Gaze point is inside the convex hull defined by the face points
   2. The Gaze point is sufficiently close ie. less than the ON_FACE_DISTANCE_THRESHOLD
   """  
   face_hull_poly = convex_hull(*face_points)
   return (face_hull_poly.encloses_point(gaze_point) or 
           face_hull_poly.distance(gaze_point) <= ON_FACE_DISTANCE_THRESHOLD)

def GetGazeFacialFeatureData(gaze_xy: Tuple[int, int], face_points: List[Tuple[int, int]]):
   """
   Identifies the closest face point, facial feature, if gaze is on facial
   region, closest facial feature distance, if gaze is on face.

   Parameters
   ----------
   gaze_xy: Tuple[int, int]
      Tuple denoting the gaze x, y coordinates
   face_points_list: List[Tuple[int, int]]
      REQUIRES ORDER TO BE SAME AS MEDIAPIPE. List of 478 Tuples of face.
   
   Returns
   -------
   Tuple[Tuple[Int, Int], String, String, Float, Bool]
      0 - Tuple denoting the closest face mesh point
      1 - Closest {right_eye, left_eye, nose, mouth} feature
      2 - Which facial feature gaze is on, 
          one of {right_eye, left_eye, nose, mouth, none}
      3 - Closest_facial_feature_dist float distance to closest feature
      4 - Indicator for if gaze is inside face or not
   """
   gaze_point       = Point(gaze_xy)
   face_points_list = list(map(Point, face_points))
   closest_point    = GetClosestPoint(gaze_point, face_points_list)
   gaze_on_face     = GazeOnFace(gaze_point, face_points_list)
   closest_feature, on_facial_feature, closest_feature_dist = GetClosestFeature(gaze_point, face_points_list)

   return closest_point, closest_feature, on_facial_feature, closest_feature_dist, gaze_on_face


def AddLandmarkInfo(gaze_file: str, mediapipe_file: str) -> pd.DataFrame:
   """
   Creates a new pandas dataframe with the additional columns described above.
   Assumes all files have timestamps corrected by Alvaro's process.

   Parameters
   ----------
   gaze_file: str
   Path to gaze file (with corrected timestamps) from glasses
   mediapipe_file: str
   Path to mediapipe file with mediapipe points for each timestep

   Returns
   -------
   pd.Dataframe
      With additional columns with an entry for each row as described above
   """
   landmarks_df   = pd.read_csv(mediapipe_file)
   gaze_df        = pd.read_csv(gaze_file)

   landmarks_df = landmarks_df.truncate(before=60, after=65)
   gaze_df = gaze_df.truncate(before=60, after=65)

   print(landmarks_df.shape, gaze_df.shape)
   
   # Initializing new columns
   # Face features
   gaze_df['closest_face_point']          = None # Tuple[int, int] or None
   gaze_df['closest_facial_feature']      = None # String or None
   gaze_df['on_facial_region']            = None # String or None
   gaze_df['closest_facial_feature_dist'] = np.Inf # Float
   gaze_df['on_face']                     = False # Bool or None
   # Body features
   gaze_df['on_body_region']              = None # String
   gaze_df['closest_body_feature_dist']   = np.Inf # Float
   gaze_df['on_body']                     = False # Bool
   
   # Iterate through each timestamp
   for (idx_row, gaze_row), (_, mediapipe_row) in zip(gaze_df.iterrows(), landmarks_df.iterrows()):
      # Check we have a gaze x,y
      print(idx_row)
      if (np.isnan(gaze_row['gaze x [px]']) or np.isnan(gaze_row['gaze y [px]'])):
         continue
      # y's are flipped in visualize step because we want to see. 
      gaze_xy = (int(gaze_row['gaze x [px]']), -1 * int(gaze_row['gaze y [px]']))
      # Get mediapipe point array
      body_points = np.array(ast.literal_eval(mediapipe_row['Body']))
      face_points = np.array(ast.literal_eval(mediapipe_row['Face']))

      if len(face_points) > 0:
         face_points[0][:, 1] = -1 * face_points[0][:, 1]
      
      # print(face_points)
      
      if len(face_points) > 0 and len(face_points[0]) > 0:
         face_landmark_data = GetGazeFacialFeatureData(gaze_xy, face_points[0])
         gaze_df.at[idx_row, 'closest_face_point']          = face_landmark_data[0]
         gaze_df.at[idx_row, 'closest_facial_feature']      = face_landmark_data[1]
         gaze_df.at[idx_row, 'on_facial_region']            = face_landmark_data[2]
         gaze_df.at[idx_row, 'closest_facial_feature_dist'] = face_landmark_data[3]
         gaze_df.at[idx_row, 'on_face']                     = face_landmark_data[4]
      
      # if len(body_points) > 0 and len(body_points[0]) > 0:
      #    body_landmark_data = GetGazeBodyFeatureData(gaze_xy, body_points[0])
      #    gaze_df.at[idx_row, 'on_body']                   = body_landmark_data[0]
      #    gaze_df.at[idx_row, 'on_body_region']            = body_landmark_data[0]
      #    gaze_df.at[idx_row, 'closest_body_feature_dist'] = body_landmark_data[0]
      # 
   # print(gaze_df)
   return gaze_df