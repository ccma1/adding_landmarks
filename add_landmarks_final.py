"""Adds Landmark information 

For a given gaze file, creates a new Pandas Dataframe with the following columns:
 - gnl_timestamp - from the MediaPipe Json
 - video_timestamp - from the MediaPipe Json
 - gaze_x - from the MediaPipe Json
 - gaze_y - from the MediaPipe Json
 - closest_point_index - index of the closest point from the array of facemesh points from the MediaPipe Json
      - None if there is no gaze point or facemesh point
 - closest_point_x - x coordinate of the closest facemesh point
      - nan if there is no gaze point or facemesh point
 - closest_point_y - y coordinate of the closest facemesh point
      - nan if there is no gaze point or facemesh point
 - closest_point_distance - distance of the gaze_xy point from the closest point
      - nan if there is no gaze point or facemesh point
 - gaze_on_face - 0 if gaze and facemesh is present but gaze isn't inside the face
      - 1 if gaze is inside the facemesh within a certain threshold
      - 2 if gaze isn't present or facemesh points aren't present
 - gaze_on_20_buffer - same as gaze_on_face except for 20% buffer defined below
 - gaze_on_30_buffer - same as gaze_on_face except for 30% buffer defined below
 - gaze_to_face_distance - distance of gaze to facemesh
      - nan if there is no gaze point or facemesh point
 - gaze_to_20_buffer_distance - same as gaze_to_face_distance except for 20% buffer
 - gaze_to_30_buffer_distance - same as gaze_to_face_distance except for 30% buffer
 - on_facial_feature - 'nan' if there is no gaze_x or no gaze_y or no facemesh 
      - or combination of {0: none, 1: left_eye, 2:right_eye, 3:nose, 4:mouth}, eg. 13 := left_eye and nose
 - closest_facial_feature - same as on_facial_feature
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

"""
This is fraction of the max(width, height) of the bounding rectangle of the convex hull.
See: https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html#shapely.Polygon.minimum_rotated_rectangle

A new column will be created for each threshold.

json takes format: 
{
   "frame_x": {
      "gnl_timestamp": "timestamp",
      "video_timestamp": "timestamp",
      "gaze_x": "",
      "gaze_y": "",
      "Body [x, y, visibility, presence]":"",
      "Face [x, y, visibility, presence]":"",
      "Hands [x, y, visibility, presence]": ""
   }
}

"""

# We use 20% and 30% of the max(width, height) where width and height are of the face convex hull.
ON_FACE_DISTANCE_THRESHOLD = [0.2, 0.3]

FACIAL_FEATURE_NAMES = ["nan", "left_eye", "right_eye", "nose", "mouth"]

# Not used for reasons described in readme
FACE_SILHOUETTE = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ]

LEFT_EYE_SILHOUETTE = [156, 70, 63, 105, 66, 107, 55, 193, 245, 128, 121, 120, 119, 118, 117, 111, 143, 156]

RIGHT_EYE_SILHOUETTE = [383, 300, 293, 334, 296, 336, 285, 417, 465, 357, 350, 349, 348, 347, 346, 340, 372, 383]

NOSE_SILHOUETTE = [8, 351, 399, 420, 429, 358, 327, 326, 2, 97, 98, 129, 209, 198, 174, 122, 8]

MOUTH_SILHOUETTE = [164, 393, 391, 322, 410, 432, 273, 335, 406, 313, 18, 83, 182, 106, 43, 212, 186, 92, 165, 167, 164]

def GetClosestPoint(gaze_point: shapely.Point, face_points: List):
   """
   Returns (index of point, (point_x, point_y), distance of gaze to point)
   We need to check all the points. See Readme for edge cases that show why.
   """
   index, (closest_point, dist) = functools.reduce(
      lambda p1, p2: p1 if p1[1][1] < p2[1][1] else p2,
      enumerate(map(lambda point: (point, math.dist(point, (gaze_point.x, gaze_point.y))), face_points))
   )

   return (index, closest_point, dist)

def GetClosestFeature(gaze_point: shapely.Point, distance_to_face: float, facial_polygons: List[shapely.Polygon]):
   """
   distance_to_face is the distance to the convex hull that defines the face.

   Returns (
      a combination of {0:none, 1: left_eye, 2:right_eye, 3:nose, 4:mouth}, eg. 13 := left_eye and nose
      a combination index of closest facial feature, one of {0:none, 1:left_eye, 2: right_eye, 3:nose, 4:mouth}, 
      distance_to_{LEFT_EYE, RIGHT_EYE, NOSE, MOUTH}, potentially equidistance features
   )
   """
   on_facial_feature, closest_facial_feature, closest_feat_dist = 0, 0, np.inf
   feature_distances = [0, 0, 0, 0]
   for i in range(4):
      feature_poly = facial_polygons[i]
      feature_distances[i] = feature_poly.distance(gaze_point)

      if (distance_to_face == 0 and feature_poly.covers(gaze_point)):
         closest_facial_feature = i + 1 if on_facial_feature == 0 else closest_facial_feature * 10 + i + 1
         on_facial_feature      = on_facial_feature * 10 + i + 1
         closest_feat_dist      = 0
         feature_distances[i]   = 0
      elif feature_distances[i] == closest_feat_dist:
         # Feature poly does not cover face point bc. If not on face then not in feature
         closest_facial_feature = closest_facial_feature * 10 + i + 1
      elif feature_distances[i] < closest_feat_dist:
         closest_facial_feature = i + 1
         closest_feat_dist = feature_distances[i]
   
   return [on_facial_feature, 
           closest_facial_feature,
           *feature_distances]

def GazeOnFace(gaze_point: shapely.Point, face_poly: shapely.Polygon): 
   """
   face_poly is a convex hull of the face.

   Returns Tuple (
      1 if gaze point is enclosed by face convex hull to face shape, or 0 otherwise,
      1 if gaze point is within the 20% buffer or 0 otherwise,
      1 if gaze point is within the 30% buffer or 0 other, 
      distance of gaze point to face convex hull,
      distance of gaze point to 20% buffer (just max(0, distance to face convex hull - 20% diameter)),
      distance of gaze point to 30% buffer (just max(0, distance to face convex hull - 20% diameter))
   )
   """  
   box_xx, box_yy = face_poly.minimum_rotated_rectangle.exterior.coords.xy
   box_points = list(zip(box_xx, box_yy))
   box_width = math.dist(box_points[1],box_points[2])
   box_height = math.dist(box_points[0],box_points[1])
   face_hull_diameter = max(box_width, box_height)

   if gaze_point.covered_by(face_poly):
      # Inside the face convex hull
      return (1, 1, 1, 0, 0, 0)
   
   # Shortest distance from gaze point to convex hull of face
   gaze_dist = face_poly.distance(gaze_point)
   if ((buff_one_dist := gaze_dist - ON_FACE_DISTANCE_THRESHOLD[0] * face_hull_diameter) <= 0):
      # Inside the 20% Buffer
      return (0, 1, 1, gaze_dist, 0, 0)
   elif ((buff_two_dist := gaze_dist - ON_FACE_DISTANCE_THRESHOLD[1] * face_hull_diameter) <= 0):
      # Inside the 30% buffer
      return (0, 0, 1, gaze_dist, buff_one_dist, 0)
   
   # Definitely outside
   return (0, 0, 0, gaze_dist, buff_one_dist, buff_two_dist)

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
   [
      # 0-3: Closest point info
      closest_point_index, 
      closest_point_x,
      closest_point_y,
      closest_point_distance,
      # 4-9: Gaze on face info
      1 if gaze point is enclosed by face convex hull to face shape, or 0 otherwise,
      1 if gaze point is within the 20% buffer or 0 otherwise,
      1 if gaze point is within the 30% buffer or 0 other, 
      distance of gaze point to face convex hull,
      distance of gaze point to 20% buffer (just max(0, distance to face convex hull - 20% diameter)),
      distance of gaze point to 30% buffer (just max(0, distance to face convex hull - 20% diameter)),
      # 10-15 Gaze on facial feature info
      a combination of {0:none, 1: left_eye, 2:right_eye, 3:nose, 4:mouth},
      a combination index of closest facial feature, one of {0:none, 1:left_eye, 2: right_eye, 3:nose, 4:mouth},
      distance_to_{LEFT_EYE, RIGHT_EYE, NOSE, MOUTH}
   ]
   """
   facial_polygons                = CreatePolygons(face_points)
   gaze_on_face_data              = GazeOnFace(gaze_xy, facial_polygons[0])
   facial_feature_data            = GetClosestFeature(gaze_xy, gaze_on_face_data[3], facial_polygons[1:])
   closest_point_index, closest_point, closest_point_dist = GetClosestPoint(gaze_xy, face_points)

   return [
      closest_point_index, 
      closest_point[0], # x-coordinate
      closest_point[1], # y-coordinate
      closest_point_dist, 
      *gaze_on_face_data,
      *facial_feature_data
   ] 

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
            frame['gaze_x'],
            frame['gaze_y'],
            np.nan,       # Index of closest face point 
            float('nan'), # Closest face point x
            float('nan'), # Closest face point y
            float('nan'), # Distance of gaze to closest face point x, y
            2,            # On face convex hull
            2,            # On 20% buffer 
            2,            # On 30% buffer 
            float('nan'), # Distance to face 
            float('nan'), # Distance to face 20%  
            float('nan'), # Distance to face 30% 
            'nan',        # On facial feature
            'nan',        # Closest facial feature
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
            [int(frame['gnl_timestamp']), int(frame['video_timestamp']), frame['gaze_x'], frame['gaze_y']] + 
            GetGazeFacialFeatureData(gaze_xy, face_lmks)
         )
         del face_lmks

   df = pd.DataFrame(result)
   df.columns = ['gnl_timestamp', 
                 'video_timestamp',
                 'gaze_x',
                 'gaze_y',
                 'closest_point_index', 
                 'closest_point_x', 
                 'closest_point_y', 
                 'closest_point_distance', 
                 'gaze_on_face',
                 'gaze_on_20_buffer',
                 'gaze_on_30_buffer',
                 'gaze_to_face_distance',
                 'gaze_to_20_buffer_distance',
                 'gaze_to_30_buffer_distance',
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
   res = AddLandmarkInfo("./data/04Apr_S1_g1_Data.json")
   res.to_csv('test_04Apr_S1_g1_Data.csv', index=False, na_rep='nan')

if __name__ == "__main__":
   import cProfile as profile
   import pstats
   import time
   # Profiling code if interested...

   PROFILE = True
   if PROFILE:
      # prof = profile.Profile()
      # prof.enable()
      # main()
      # prof.disable()
      # # print program output
      # print('Done!')
      # # print profiling output
      # stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
      # stats.print_stats(30) 
      start = time.time()
      main()
      print(time.time() - start)
   else: 
      main()