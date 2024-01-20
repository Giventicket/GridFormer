import cv2
import numpy as np
import torch

class TSPImageDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, img_size, point_radius=2, point_color=1, line_thickness=2, line_color=0.5, max_points=100):
    self.data_file = data_file
    self.img_size = img_size
    self.point_radius = point_radius
    self.point_color = point_color
    self.line_thickness = line_thickness
    self.line_color = line_color
    self.max_points = max_points

    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_reordered_tsp_instance(self, points, tour):
    point2before_idx = {}
    for idx, point in enumerate(points):
        point2before_idx[tuple(point)] = idx + 1

    center_of_mass = np.mean(points, axis=0)
    coordinates = np.array(points)
    centered_coordinates = coordinates - center_of_mass
    angles = np.arctan2(centered_coordinates[:, 1], centered_coordinates[:, 0])
    scaled_angles = (angles + 2 * np.pi) % (2 * np.pi)
    distances = np.sqrt(centered_coordinates[:, 0] ** 2 + centered_coordinates[:, 1] ** 2)
    reordered_points = coordinates[np.lexsort((distances, scaled_angles))] # 증가하는 순서대로
    
    before_idx2after_idx = {}
    for idx, point in enumerate(reordered_points):
        before_idx = point2before_idx[tuple(point)]
        before_idx2after_idx[before_idx] = idx + 1
    
    after_tour = []
    for before_idx in tour:
        after_idx = before_idx2after_idx[before_idx]
        after_tour.append(after_idx)
        
    after_tour = np.array(after_tour[:-1])
    one_index = np.where(after_tour == 1)[0][0]
    one_before = after_tour[(one_index - 1 + len(after_tour)) % len(after_tour)]
    one_after = after_tour[(one_index + 1 + len(after_tour)) % len(after_tour)]

    if one_after < one_before:
        reordered_tour = np.roll(after_tour, -one_index)
    else:
        reordered_tour = np.roll(after_tour, len(after_tour) - one_index - 1)[::-1]
    
    reordered_tour = np.append(reordered_tour, 1)
        
    return reordered_points, reordered_tour, center_of_mass

  def rasterize(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    
    # Rasterize lines
    img = np.zeros((self.img_size, self.img_size))
    
    points, tour, center_of_mass = self.get_reordered_tsp_instance(points, tour) # reorder points, tour
    cv2.circle(img, ((self.img_size - 1) * center_of_mass).astype(int), radius=self.point_radius, color=self.point_color, thickness=-1)
    
    for i in range(tour.shape[0] - 1):
      from_idx = tour[i] - 1
      to_idx = tour[i + 1] - 1

      cv2.line(img,
               ((self.img_size - 1) * points[from_idx, ::-1]).astype(int),
               ((self.img_size - 1) * points[to_idx, ::-1]).astype(int),
               color=self.line_color, thickness=self.line_thickness)

    for i in range(len(tour)):
      point = ((self.img_size - 1) * points[tour[i] - 1, ::-1]).astype(int)
      cv2.circle(img, point, radius=self.point_radius, color=self.point_color, thickness=-1)
      text_position = (point[0] - 10, point[1] - 10)
      cv2.putText(img, str(tour[i]), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)  

    # Rescale image to [-1,1]
    img = 2 * (img - 0.5)

    return img, points, tour

  def __getitem__(self, idx):
    img, points, tour = self.rasterize(idx)
    return img, points, tour


if __name__ == "__main__":
    images = TSPImageDataset(
        data_file = "tsp20-20_concorde.txt", 
        img_size = 640, 
        point_radius = 5, 
        point_color = 250, 
        line_thickness = 2,
        line_color = 200,
        max_points = 100
    )
    
    for idx in range(len(images)):
        img, points, tour = images[idx]
        print(idx, tour)