"""Adds Landmark information 

For a given gaze file, creates a new Pandas Dataframe with the following columns:
 - gnl_timestamp - from the MediaPipe Json
 - video_timestamp - from the MediaPipe Json
 - gaze_timestamp - from the MediaPipe Json
 - closest_point_index - index of the closest point from the array of facemesh points from the MediaPipe Json
      - None if there is no gaze point or facemesh point
 - closest_point_x - x coordinate of the closest facemesh point
      - nan if there is no gaze point or facemesh point
 - closest_point_y - y coordinate of the closest facemesh point
      - nan if there is no gaze point or facemesh point
 - closest_point_distance - distance of the gaze_xy point from the closest point
      - nan if there is no gaze point or facemesh point
 - on_face - 0 if gaze and facemesh is present but gaze isn't inside the face
      - 1 if gaze is inside the facemesh within a certain threshold
      - 2 if gaze isn't present or facemesh points aren't present
 - distance_to_face - distance of gaze to facemesh
      - nan if there is no gaze point or facemesh point
 - on_facial_feature - if gaze point is inside polygon of facial feature.
      - one of {left_eye, right_eye, nose, mouth, none}
 - closest_facial_feature - the closest facial feature to the gaze point
      - one of {left_eye, right_eye, nose, mouth}
 - distance_to_left_eye - distance to left eye
      - nan if there is no gaze point or facemesh point
 - distance_to_right_eye - distance to right eye
      - nan if there is no gaze point or facemesh point
 - distance_to_nose_eye - distance to nose eye
      - nan if there is no gaze point or facemesh point
 - distance_to_mouth_eye - distance to mouth eye   
      - nan if there is no gaze point or facemesh point

See Readme for additional details regarding algorithms. 
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import csv
import functools
import json
import time
import math
import shapely

from typing import Tuple, List
from shapely import Point, Polygon, MultiPoint
from operator import itemgetter

ON_FACE_DISTANCE_THRESHOLD = 0

FACIAL_FEATURE_NAMES = ["left_eye", "right_eye", "nose", "mouth", "none"]

FACE_SILHOUETTE = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ]

LEFT_EYE_SILHOUETTE = [156, 70, 63, 105, 66, 107, 55, 193, 245, 128, 121, 120, 119, 118, 117, 111, 143, 156]

RIGHT_EYE_SILHOUETTE = [383, 300, 293, 334, 296, 336, 285, 417, 465, 357, 350, 349, 348, 347, 346, 340, 372, 383]

NOSE_SILHOUETTE = [8, 351, 399, 420, 429, 358, 327, 326, 2, 97, 98, 129, 209, 198, 174, 122, 8]

MOUTH_SILHOUETTE = [164, 393, 391, 322, 410, 432, 273, 335, 406, 313, 18, 83, 182, 106, 43, 212, 186, 92, 165, 167, 164]

NON_FEATURE_POINTS = [9, 10, 21, 32, 34, 36, 47, 50, 54, 58, 67, 68, 69, 71, 93, 100, 101, 103, 104, 108, 109, 114, 116, 123, 126, 127, 132, 135, 136, 137, 138, 139, 140, 142, 147, 148, 149, 150, 151, 152, 162, 169, 170, 171, 172, 175, 176, 177, 187, 188, 192, 194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 213, 214, 215, 216, 217, 227, 234, 251, 262, 264, 266, 277, 280, 284, 288, 297, 298, 299, 301, 323, 329, 330, 332, 333, 337, 338, 343, 345, 352, 355, 356, 361, 364, 365, 366, 367, 368, 369, 371, 376, 377, 378, 379, 389, 394, 395, 396, 397, 400, 401, 411, 412, 416, 418, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 433, 434, 435, 436, 437, 447, 454]

def GetClosestPoint(gaze_point: shapely.Point, face_points: List):
   """
   Returns (index of point, (point_x, point_y), distance of gaze to point)
   """
   closest_point = functools.reduce(
        lambda p1, p2: p1 if math.dist((gaze_point.x, gaze_point.y),p1[1]) <= math.dist((gaze_point.x, gaze_point.y),p2[1]) else p2,
        list(enumerate(face_points))
    )
   
   return (closest_point[0], closest_point[1], math.dist((gaze_point.x, gaze_point.y), closest_point[1]))

def GetClosestFeature(gaze_point: shapely.Point, distance_to_face: float, facial_polygons: List[shapely.Polygon]):
   """
   Returns (
      index of facial feature gaze is on, one of {0:left_eye, 1: right_eye, 2:nose, 3:mouth, 4:none},
      index of closest facial feature, one of {0:left_eye, 1: right_eye, 2:nose, 3:mouth, 4:none}, 
      distance_to_{LEFT_EYE, RIGHT_EYE, NOSE, MOUTH}
   )
   """
   on_facial_feature, closest_facial_feature, closest_feat_dist = 4, None, None
   feature_distances = [0, 0, 0, 0]
   for i in range(4):
      feature_poly = facial_polygons[i]
      feature_distances[i] = feature_poly.distance(gaze_point)

      if (distance_to_face == 0 and feature_poly.contains(gaze_point)):
         on_facial_feature, closest_facial_feature, closest_feat_dist = i, i, 0
         feature_distances[i] = 0
      elif closest_feat_dist == None or feature_distances[i] < closest_feat_dist:
            closest_facial_feature, closest_feat_dist = i, 0
   
   return [on_facial_feature, 
           closest_facial_feature,
           feature_distances[0],
           feature_distances[1],
           feature_distances[2],
           feature_distances[3]]

def GazeOnFace(gaze_point: shapely.Point, face_poly: shapely.Polygon): 
   """
   Returns Tuple (
      1 if gaze point is within ON_FACE_DISTANCE_THRESHOLD or enclosed by 
      face polygon to face shape, or 0 otherwise,
      distance of gaze point to face polygon
   )
   """  
   if face_poly.contains(gaze_point):
      return (1, 0)
   
   gaze_dist = face_poly.distance(gaze_point)
   return (1 if gaze_dist <= ON_FACE_DISTANCE_THRESHOLD else 0, gaze_dist)

def CreatePolygons(face_points: List[Tuple[float, float]]):
   """
   Returns shapely Polygons defining the silhouette for the face, eyes, nose, mouth
   """
   return [
      MultiPoint(face_points).convex_hull,
      Polygon(itemgetter(*LEFT_EYE_SILHOUETTE)(face_points)),
      Polygon(itemgetter(*RIGHT_EYE_SILHOUETTE)(face_points)),
      Polygon(itemgetter(*NOSE_SILHOUETTE)(face_points)),
      Polygon(itemgetter(*MOUTH_SILHOUETTE)(face_points))
   ]
   
def GetGazeFacialFeatureData(gaze_xy: shapely.Point, face_points: List):
   """
   Identifies the closest face point, facial feature, if gaze is on facial
   region, closest facial feature distance, if gaze is on face.

   Parameters
   ----------
   gaze_xy: 
   face_points_list: 
   
   Returns
   -------
   [closest_point_index, 
    closest_point: (x, y), 
    gaze_on_face, 
    distance_to_face,
    on_facial_feature,  # {left_eye, right_eye, nose, mouth, none}
    closest_facial_feature, # {left_eye, right_eye, nose, mouth}
    left_eye_distance,
    right_eye_distance,
    nose_distance,
    mouth_distance]
   """
   facial_polygons                = CreatePolygons(face_points)
   gaze_on_face, distance_to_face = GazeOnFace(gaze_xy, facial_polygons[0])
   facial_feature_data            = GetClosestFeature(gaze_xy, distance_to_face, facial_polygons[1:])
   if gaze_on_face and facial_feature_data[0] == 4:
      # Gaze on face but not on any facial feature
      closest_point_index, closest_point, closest_point_dist = GetClosestPoint(gaze_xy, face_points)
   elif gaze_on_face and facial_feature_data[0] != 4:
      # Gaze on face and on facial feature
      closest_point_index, closest_point, closest_point_dist = GetClosestPoint(gaze_xy, face_points)
   else:
      # Gaze not on face
      closest_point_index, closest_point, closest_point_dist = GetClosestPoint(gaze_xy, face_points)
   return [
      closest_point_index, 
      closest_point[0], 
      closest_point[1], 
      closest_point_dist, 
      gaze_on_face, 
      distance_to_face,
      FACIAL_FEATURE_NAMES[facial_feature_data[0]], # Mapping to naps
      FACIAL_FEATURE_NAMES[facial_feature_data[1]]
   ] + facial_feature_data[2:]

def AddLandmarkInfo(mediapipe_file: str) -> pd.DataFrame:
   """
   Creates a new pandas dataframe with the additional columns described above.
   Assumes all files have timestamps corrected by Alvaro's process.

   Parameters
   ----------
   mediapipe_file: str
   Path to JSON mediapipe file with mediapipe points for each timestep

   Returns
   -------
   pd.Dataframe
   """
   mediapipe_data = json.load(open(mediapipe_file))
   result         = []
   for frame in mediapipe_data.values():

      if (frame['gaze_x'] == 'nan' or 
          frame['gaze_y'] == 'nan' or 
          frame['Face [x, y, visibility, presence]'] == ""):
         result.append([
            int(frame['gnl_timestamp']), 
            int(frame['video_timestamp']),
            int(frame['gaze_timestamp']),
            None,         # Index of closest face point 
            float('nan'), # Closest face point x
            float('nan'), # Closest face point y
            float('nan'), # Distance of gaze to closest face point x, y
            2,            # On face
            float('nan'), # Distance to face 
            'none',       # On facial feature
            'none',       # Closest facial feature
            float('nan'), # Distance to left_eye
            float('nan'), # Distance to right_eye
            float('nan'), # Distance to nose
            float('nan'), # Distance to mouth
         ])
      else:
         gaze_xy = Point(float(frame['gaze_x']), float(frame['gaze_y']))
         face_lmks = [
            (float((lm.split(',')[:2])[0]), float(float((lm.split(',')[:2])[1])))
            for lm in frame['Face [x, y, visibility, presence]'].split()
         ]
         result.append(
            [int(frame['gnl_timestamp']), int(frame['video_timestamp']),int(frame['gaze_timestamp'])] + 
            GetGazeFacialFeatureData(gaze_xy, face_lmks)
         )
         del face_lmks

   df = pd.DataFrame(result)
   df.columns = ['gnl_timestamp', 
                 'video_timestamp', 
                 'gaze_timestamp', 
                 'closest_point_index', 
                 'closest_point_x', 
                 'closest_point_y', 
                 'closest_point_distance', 
                 'gaze_on_face',
                 'gaze_to_face_distance',
                 'on_facial_feature',
                 'closest_facial_feature', 
                 'gaze_to_left_eye_distance',
                 'gaze_to_right_eye_distance',
                 'gaze_to_nose_distance',
                 'gaze_to_mouth_distance'
                 ]
   
   df.sort_values(by='gnl_timestamp', ascending=True)
   return df

def main():
   # res = AddLandmarkInfo("./data/test_data.json")
   # res = AddLandmarkInfo("./data/20Mar_S1_g2_Data.json")
   res = AddLandmarkInfo("./data/04Apr_S1_g2_Data.json")
   res.to_csv('test_04Apr_S1_g2_Data.csv', index=False, na_rep='nan')

if __name__ == "__main__":
   import cProfile as profile
   import pstats
   prof = profile.Profile()
   prof.enable()
   main()
   prof.disable()
   # print program output
   print('Done!')
   # print profiling output
   stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
   stats.print_stats(30) 