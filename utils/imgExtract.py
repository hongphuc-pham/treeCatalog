from PIL import Image
import os, time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


DATASET_DIRECTORY = '/content/drive/MyDrive/Uni/Thesis/dataset/detection/PasadenaUrbanTrees'
# Radius in meters of Earth
EARTH_RADIUS = 6371000  # in meters
STREET_THRESHOLD = 626 # in pixel (height top to bottom)


class AerialExtractor:
  def __init__(self, dataset_dir):
    """
    Constructor of helper class for extracting aerial images from geographical locations, 
    :param dataset_dir (str): location of the PasadenaUrbanTrees folder
    :return:
    """
    self.dataset_dir = dataset_dir
    
    # Zoom parameter used to download aerial images
    self.aerial_zoom = 20 
    
    # width/height dimension of aerial tile images
    self.aerial_tile_size = 256  
  
  def extract_image(self, lat, lng, width, height, output_image_name=None):
    """
    Extract an aerial image around a given latitude,longitude location
    :param lat (float): latitude
    :param lng (float): longitude
    :param width (int): width in pixels of the extracted image
    :param height (int): height in pixels of the extracted image
    :param output_image_name (str): If not None, save the extracted image to this file name
    :return: im (Image): the extracted image
    """
    # map/tile coordinates of the center of the object loc
    tile_x, tile_y, pix_x, pix_y = self.geo_coords_to_tile_coords(lat, lng)
  
    # Geographic coordinates of the upper left and lower right coordinates of the box 
    # around the location
    tx1, ty1, px1, py1 = self.fix_tile_coords(tile_x, tile_y, 
                                              pix_x-width/2, pix_y-height/2)
    tx2, ty2, px2, py2 = self.fix_tile_coords(tile_x, tile_y, 
                                              pix_x+width/2, pix_y+height/2)

    # Aerial map images are divided into many tile images.  Paste the relevant tiles together
    # to extract the region around the location
    im = Image.new("RGB", ((tx2-tx1+1)*self.aerial_tile_size, 
                           (ty2-ty1+1)*self.aerial_tile_size), "black")
    for i in range(tx1, tx2+1):
      for j in range(ty1, ty2+1):
        fname = os.path.join(self.dataset_dir, 'aerial', 'x' + str(i) + '_y' + 
                             str(j) + '_z' + str(self.aerial_zoom) + ".png")
        tile = Image.open(fname)
        im.paste(tile, ((i-tx1)*self.aerial_tile_size, (j-ty1)*self.aerial_tile_size))
    im = im.crop((px1, py1, (tx2-tx1)*self.aerial_tile_size+px2+1, 
                  (ty2-ty1)*self.aerial_tile_size+py2+1))
    
    # Save the image to disk, if necessary
    if not output_image_name is None:
      im.save(output_image_name)
    
    return im


  def geo_coords_to_tile_coords(self, lat, lng):
    """ 
    Convert a latitude, longitude coordinate to the appropriate tile xy-coordinate in 
    aerial map imagery.  The surface of the earth is flattened out into a giant map
    image and then separated into tile images.  See 
    https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    :param lat (float): Latitude
    :param lng (float): Longitude
    :return: tile_x (int), tile_y (int), pix_x (int), pix_y (int): The aerial tile image 
    containing lat,lng is tile tile_x,tile_y. It's at pixel location (pix_x,pix_y) in the 
    image
    """
    scale = 1 << self.aerial_zoom
    siny = math.sin(lat * math.pi / 180)
    
    # Truncating to 0.9999 effectively limits latitude to 89.189. This is
    # about a third of a tile past the edge of the world tile.
    siny = min(max(siny, -0.9999), 0.9999)
    
    x = self.aerial_tile_size * (0.5 + lng / 360)
    y = self.aerial_tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    tile_x = math.floor(x*scale / self.aerial_tile_size)
    tile_y = math.floor(y*scale / self.aerial_tile_size)
    pixel_x = math.floor(x*scale) % self.aerial_tile_size
    pixel_y = math.floor(y*scale) % self.aerial_tile_size
    
    return (int(tile_x), int(tile_y), int(pixel_x), int(pixel_y))

  def tile_coords_to_geo_coords(self, tile_x, tile_y, pixel_x, pixel_y):
    """ 
    Convert a pixel location in an aerial tile image to a latitude, longitude.  See
    https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    :param tile_x (int): The tile id in the x direction
    :param tile_y (int): The tile id in the y direction
    :param pix_x (int): x-pixel coordinate in the tile image
    :param pix_y (int): y-pixel coordinate in the tile image
    :return: lat (float), lng (float): corresponding latitude, longitude
    """
    scale = float(1 << self.aerial_zoom)
    x = (tile_x*self.aerial_tile_size + pixel_x) / scale
    y = (tile_y*self.aerial_tile_size + pixel_y) / scale
    k = math.exp(-(y / self.aerial_tile_size - 0.5) * (4 * math.pi))
    lng = (x / self.aerial_tile_size - 0.5) * 360.0
    lat = math.asin((k-1) / (1+k)) * 180.0 / math.pi
    return lat, lng

  def fix_tile_coords(self, tile_x, tile_y, pix_x, pix_y):
    """ 
    Helper function to correct tile/pixel coordinates. If a pixel location places it outside
    the boundary of the tile, it moves to the appropriate tile image
    """
    tile_x, pix_x = self.fix_tile_coords_x(tile_x, pix_x)
    tile_y, pix_y = self.fix_tile_coords_x(tile_y, pix_y)
    return tile_x, tile_y, pix_x, pix_y

  def fix_tile_coords_x(self, tile_x, pix_x):
    """ 
    Helper function to correct tile/pixel coordinates. If a pixel location places it outside
    the boundary of the tile, it moves to the appropriate tile image.  This helper function
    corrects a single coordinate (x or y), rather than both
    """
    while pix_x < 0:
      tile_x -= 1
      pix_x += self.aerial_tile_size
    while pix_x >= self.aerial_tile_size:
      tile_x += 1
      pix_x -= self.aerial_tile_size
    return tile_x, pix_x
  
  

class StreetViewExtractor:
  def __init__(self, dataset_dir, panos):
    """
    Constructor of helper class for extracting street view images from geographical 
    locations.  This class does not undistort/rectify the panorama
    :param dataset_dir (str): location of the PasadenaUrbanTrees folder
    :param panos (dict): Meta-data of all street view panorama locations
    :return:
    """
    self.dataset_dir = dataset_dir
    self.panos = panos
    self.earth_radius = 6371000

    # Zoom parameter used to download street view panorama images
    self.streetview_zoom = 2 
    
    # height in meters that cameras are mounted on Google streetview cars
    self.google_car_camera_height = 3.0 
  
  def get_nearest_pano(self, lat, lng, imgShow = None):
    """
    Find the closest street view panorama to a given latitude/longitude
    :param lat (float): latitude
    :param lng (float): longitude
    :imgShow: None : showing the image or not
    :return: pano (dict): meta-data of the closest panorama
    """
    best_d = 10000000
    pano = None
    for pid in self.panos:
      lat1 = float(self.panos[pid]['Location']['original_lat'])
      lng1 = float(self.panos[pid]['Location']['original_lng'])
      dist = haversine_distance(lat, lng, lat1, lng1)
      if dist < best_d:
        pano = self.panos[pid]
        best_d = dist

    # show the panorama image
    if imgShow is not None:
      full_pano_image = Image.open(os.path.join(self.dataset_dir, "streetview", 
                                                pano['Location']['panoId'] + "_z2.jpg"))
      plt.figure(figsize=(18,10))
      plt.axis('off')
      plt.imshow(full_pano_image)

    return pano
  
  def extract_image(self, pano, lat, lng, object_dims, height = 0, 
                    full_pano_image = None, output_image_name = None, 
                    imgShow = None):
    """
    For a given street view panorama, suppose we want to look at an object at geographic
    location lat, lng at a given height above the ground plane.  Return a cropped image
    of that object that is extracted from the appropriate region of the panorama image.
    This function does not undistort/rectify the panorama
    :param pano (dict): meta-data for the street view panorama location
    :param lat (float): latitude
    :param lng (float): longitude
    :param height (float): height in meters of the object above the ground plane
    :param object_dims (width (float), height (float)): width,height of the object in meters
    :param full_pano_image (Image): the full panorama image from which to crop the object.
              If None, then the panorama image is loaded from disk
    :param output_image_name (str): If not None, save the extracted image to this file name
    :imgShow: None : showing the image  with bounding box
    :return: image (Image): extracted image of the object
    """
    
    # Compute the bounding box in the pano image containing the object we want to lookat
    max_zoom = int(pano['Location']['zoomLevels'])
    down = int(math.pow(2,max_zoom-self.streetview_zoom))  # downsample amount
    image_width = int(pano['Data']['image_width'])//down
    image_height = int(pano['Data']['image_height'])//down
    x1, y1, x2, y2 = self.geo_coords_to_streetview_bbox(pano, lat, lng, object_dims, 
                                                        height=height)
    
    # Load the full pano image from disk, if it isn't passed in as an argument
    if full_pano_image is None:
      full_pano_image = Image.open(os.path.join(self.dataset_dir, "streetview", 
                                                pano['Location']['panoId'] + "_z2.jpg"))
    
    
    # Crop out the bounding box, padding the image horizontally since it wraps around 
    # 360 degrees
    
    padded = Image.new("RGB", (int(image_width*2), int(image_height*2)), "black")
    px, py = image_width//2, image_height//2
    padded.paste(full_pano_image.crop((image_width//2,0,image_width,image_height)), (0,py))
    padded.paste(full_pano_image, (px,py))
    padded.paste(full_pano_image.crop((0,0,image_width//2,image_height)), (px+image_width,py))
    image = padded.crop((int(x1+px), int(y1+py), int(x2+px), int(y2+py))) 
    
    # Save the image to disk, if necessary
    if not output_image_name is None:
      image.save(output_image_name)
    
    # show the panorama image with bounding box
    if imgShow is not None:
      pano_image = Image.open(os.path.join(self.dataset_dir, "streetview", 
                                                pano['Location']['panoId'] + "_z2.jpg"))
      img = np.array(pano_image)
      print(f'{x1} - {y1} - {x2} - {y2}')
      cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,120,212),3)
      plt.figure(figsize=(18,10))
      plt.axis('off')
      plt.imshow(img)

    return image

  def geo_coords_to_streetview_pixel(self, pano, lat, lng, height=0, object_dims=None):
    """
    For a given street view panorama, suppose we want to look at an object at geographic
    location lat, lng at a given height above the ground plane.  Return the appropriate
    streetview image location
    :param pano (dict): meta-data for the street view panorama location
    :param lat (float): latitude
    :param lng (float): longitude
    :param height (float): height in meters of the object above the ground plane
    :return: x (float), y (float) pixel location in the streetview image
    """
    
    max_zoom = int(pano['Location']['zoomLevels'])
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg'])*math.pi/180
    lat1 = float(pano['Location']['original_lat'])
    lng1 = float(pano['Location']['original_lng'])

    # Suppose the camera is at x=0, y=0, and the y-axis points north, and the x-axis points 
    # east. Compute the position (dx,dy) corresponding to (lat,lng).  
    # dy=math.sin(lat-lat1)*EARTH_RADIUS.  dx=math.cos(lat1)*math.sin(lng-lng1)*EARTH_RADIUS.
    # math.atan2(dx, dy) is then the clockwise angle from north, which sits in the center of 
    # the panorama image (if the panorama image is first shifted by yaw degrees, such that 
    # north is in the center of the image).
    dx = math.cos(math.radians(lat1))*math.sin(math.radians(lng-lng1))
    dy = math.sin(math.radians(lat-lat1))
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw  
    while look_at_angle > 2*math.pi: look_at_angle = look_at_angle-2*math.pi
    while look_at_angle < 0: look_at_angle = look_at_angle+2*math.pi
    z = math.sqrt(dx*dx+dy*dy)*EARTH_RADIUS

    down = int(math.pow(2,max_zoom-self.streetview_zoom))  # downsample amount
    image_width = int(pano['Data']['image_width'])/down
    image_height = int(pano['Data']['image_height'])/down
    
    # Return the streetview pixel location corresponding to lat,lng
    x = (image_width*look_at_angle)/(2*math.pi)
    y = image_height/2 - image_height*(math.atan2(height-self.google_car_camera_height, z)
                                       - pitch)/(math.pi) 
    return x, y 

  def streetview_pixel_to_geo_coords(self, pano, x, y, height=0, zoom=None):
    """
    For a given street view panorama image, suppose we are looking at pixel x,y and want
    want to figure out the corresponding geographic location lat,lng under the assumption
    that the point we're looking at is at a known height above the ground plane.  Return 
    the appropriate latitude, longitude
    :param pano (dict): meta-data for the street view panorama location
    :param x (float): x-pixel coordinate
    :param y (float): y-pixel coordinate
    :param height (float): height in meters of the object above the ground plane
    :param object_dims (width,height): The dimensions (in meters) of the object of interest.
       If defined, we return a bounding box in the image rather than a pixel location
    :return: x (float), y (float) pixel location in the streetview image
    """
    
    max_zoom = int(pano['Location']['zoomLevels'])
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg'])*math.pi/180
    lat1 = float(pano['Location']['original_lat'])
    lng1 = float(pano['Location']['original_lng'])
    
    down = int(math.pow(2,max_zoom-self.streetview_zoom))  # downsample amount
    image_width = int(pano['Data']['image_width'])/down
    image_height = int(pano['Data']['image_height'])/down

    look_at_angle = x*(2*math.pi)/image_width
    tilt_angle = (image_height/2-y)*math.pi/image_height+pitch
    z = (height-self.google_car_camera_height) / math.tan(min(-1e-2,tilt_angle))
    dx = math.sin(look_at_angle-math.pi+yaw)*z/self.earth_radius
    dy = math.cos(look_at_angle-math.pi+yaw)*z/self.earth_radius
    lat = lat1 + math.degrees(math.asin(dy))
    lng = lng1 + math.degrees(math.asin(dx/math.cos(math.radians(lat1))))
    
    return lat,lng
  
  def geo_coords_to_streetview_bbox(self, pano, lat, lng, object_dims, height=0):
    """
    For a given street view panorama, suppose we want to look at an object at geographic
    location lat, lng at a given height above the ground plane.  Return the appropriate
    streetview bounding box location
    :param pano (dict): meta-data for the street view panorama location
    :param lat (float): latitude
    :param lng (float): longitude
    :param height (float): height in meters of the object above the ground plane
    :param object_dims (width,height): The dimensions (in meters) of the object of interest.
    :return: x1 (float), y1 (float), x2 (float), y2 (float): bounding box in the streetview
    image
    """
    
    max_zoom = int(pano['Location']['zoomLevels'])
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg'])*math.pi/180
    lat1 = float(pano['Location']['original_lat'])
    lng1 = float(pano['Location']['original_lng'])

    # Suppose the camera is at x=0, y=0, and the y-axis points north, and the x-axis points 
    # east. Compute the position (dx,dy) corresponding to (lat,lng).  
    # dy=math.sin(lat-lat1)*EARTH_RADIUS.  dx=math.cos(lat1)*math.sin(lng-lng1)*EARTH_RADIUS.
    # math.atan2(dx, dy) is then the clockwise angle from north, which sits in the center of 
    # the panorama image (if the panorama image is first shifted by yaw degrees, such that 
    # north is in the center of the image).
    dx = math.cos(math.radians(lat1))*math.sin(math.radians(lng-lng1))
    dy = math.sin(math.radians(lat-lat1))
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw  
    while look_at_angle > 2*math.pi: look_at_angle = look_at_angle-2*math.pi
    while look_at_angle < 0: look_at_angle = look_at_angle+2*math.pi
    z = math.sqrt(dx*dx+dy*dy)*EARTH_RADIUS

    down = int(math.pow(2,max_zoom-self.streetview_zoom))  # downsample amount
    image_width = int(pano['Data']['image_width'])/down
    image_height = int(pano['Data']['image_height'])/down
    
    # Return a bounding box around the appropriate location in a streetview pixel 
    # corresponding to lat,lng
    x1 = image_width*(math.atan2(-object_dims[0]/2, z)+look_at_angle)/(2*math.pi)
    x2 = image_width*(math.atan2(object_dims[0]/2, z)+look_at_angle)/(2*math.pi)
    y1 = image_height/2 - image_height/math.pi*(
      math.atan2(height + object_dims[1]-self.google_car_camera_height, z) + pitch)
    y2 = image_height/2 - image_height/math.pi*(
      math.atan2(height-self.google_car_camera_height, z) + pitch)
    return x1, y1, x2, y2

  def geoCoords_to_sViewFov_bbox(self, pano, lat, lng, fov=80, object_dims = [0,0], height=0):
    """
    For a given street view panorama, suppose we want to look at an object at geographic
    location lat, lng at a given height above the ground plane.  Return the appropriate
    streetview bounding box location
    :param pano (dict): meta-data for the street view panorama location
    :param lat (float): latitude
    :param lng (float): longitude
    :param height (float): height in meters of the object above the ground plane
    :param fov (int): field of view, in this dataset will have three values - 40, 80 (default), 110.
    :param object_dims (width,height): The dimensions (in meters) of the object of interest.
    :return: x1 (float), y1 (float), x2 (float), y2 (float): bounding box in the streetview
    image
    """
    
    max_zoom = int(pano['Location']['zoomLevels'])
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg'])*math.pi/180
    lat1 = float(pano['Location']['original_lat'])
    lng1 = float(pano['Location']['original_lng'])

    # Suppose the camera is at x=0, y=0, and the y-axis points north, and the x-axis points 
    # east. Compute the position (dx,dy) corresponding to (lat,lng).  
    # dy=math.sin(lat-lat1)*EARTH_RADIUS.  dx=math.cos(lat1)*math.sin(lng-lng1)*EARTH_RADIUS.
    # math.atan2(dx, dy) is then the clockwise angle from north, which sits in the center of 
    # the panorama image (if the panorama image is first shifted by yaw degrees, such that 
    # north is in the center of the image).
    dx = math.cos(math.radians(lat1))*math.sin(math.radians(lng-lng1))
    dy = math.sin(math.radians(lat-lat1))
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw  
    while look_at_angle > 2*math.pi: look_at_angle = look_at_angle-2*math.pi
    while look_at_angle < 0: look_at_angle = look_at_angle+2*math.pi
    z = math.sqrt(dx*dx+dy*dy)*EARTH_RADIUS

    down = int(math.pow(2,max_zoom-self.streetview_zoom))  # downsample amount
    image_width = int(pano['Data']['image_width'] * np.radians(fov) /(2 * down * math.pi))
    image_height = int(pano['Data']['image_height'] * np.radians(fov) /(2 * down * math.pi))
    
    # Return a bounding box around the appropriate location in a streetview pixel 
    # corresponding to lat,lng
    x1 = image_width*(math.atan2(-object_dims[0]/2, z)+look_at_angle)/(2*math.pi)
    x2 = image_width*(math.atan2(object_dims[0]/2, z)+look_at_angle)/(2*math.pi)
    y1 = image_height/2 - image_height/math.pi*(
      math.atan2(height + object_dims[1]-self.google_car_camera_height, z) + pitch)
    y2 = image_height/2 - image_height/math.pi*(
      math.atan2(height-self.google_car_camera_height, z) + pitch)
    
    return x1, y1, x2, y2


def haversine_distance(lat1, lng1, lat2, lng2):
  """
  Compute the shortest path curved distance between 2 points (lat1,lng1) and 
  (lat2,lng2) using the Haversine formula.  
  """
  hlat = math.sin(math.radians((lat2-lat1)/2.0))**2
  hlng = math.sin(math.radians((lng2-lng1)/2.0))**2
  a = hlat + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*hlng
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  return EARTH_RADIUS*c

def array_haversine_distances(lat, lng, a, max_dist=None):
  if not max_dist is None:
    k = math.tan(max_dist/(EARTH_RADIUS*2.0))**2
    k2 = math.sqrt(k/(1+k))
    dlat = math.asin(k2)*360.0/math.pi
    dlng = math.asin(k2/math.cos(math.radians(lat)))*360.0/math.pi
    inds = np.argwhere(np.all(np.vstack((np.abs(a[:,0]-lat) < dlat, 
                       np.abs(a[:,1]-lng) < dlng)), axis=0))
    aa = (np.square(np.sin((a[inds,0]-lat)*math.pi/360.0)) + 
          (math.cos(math.radians(lat))*np.cos(a[inds,0]*math.pi/180.0))*
           np.square(np.sin((a[inds,1]-lng)*math.pi/360.0)))
    dists = np.empty((a.shape[0]))
    dists.fill(max_dist)
    dists[inds] = 2 * EARTH_RADIUS * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))
    return dists
  else:
    aa = (np.square(np.sin((a[:,0]-lat)*math.pi/360.0)) + 
          (math.cos(math.radians(lat))*np.cos(a[:,0]*math.pi/180.0))*
           np.square(np.sin((a[:,1]-lng)*math.pi/360.0)))
    dists = 2 * EARTH_RADIUS * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))
  return dists
