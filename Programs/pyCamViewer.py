import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import json

# Constants for camera FOV
CAMERA_FOV = 114  # degrees
CAMERA_RANGE = 500  # units
NUM_RAYS = 100

def calculate_coverage(camera_pos, direction, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles):
    fov_points = [camera_pos]
    start_angle_deg = direction - CAMERA_FOV / 2
    end_angle_deg = direction + CAMERA_FOV / 2
    for angle_deg in np.linspace(start_angle_deg, end_angle_deg, NUM_RAYS):
        angle_rad = np.radians(angle_deg)  # Convert angle to radians
        ray_end = (camera_pos[0] + np.cos(angle_rad) * CAMERA_RANGE, 
                   camera_pos[1] + np.sin(angle_rad) * CAMERA_RANGE)
        ray = LineString([camera_pos, ray_end])
        trimmed_ray = trim_ray_at_obstacle(ray, obstacles)
        fov_points.append(trimmed_ray.coords[-1])
    return Polygon(fov_points)

def trim_ray_at_obstacle(ray, obstacles):
    intersections = [ray.intersection(obstacle) for obstacle in obstacles if ray.intersects(obstacle)]
    if not intersections:
        return ray

    # Find the closest intersection point to the start of the ray
    closest_intersection = min(intersections, key=lambda inter: Point(ray.coords[0]).distance(inter))

    # Handle different types of geometries
    if closest_intersection.geom_type == 'Point':
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    elif closest_intersection.geom_type == 'LineString':
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    elif closest_intersection.geom_type == 'MultiLineString':
        # For MultiLineString, find the closest point among all line strings
        closest_point = None
        min_distance = float('inf')
        for line in closest_intersection.geoms:  # Iterate over each LineString in the MultiLineString
            point = line.coords[0]  # Taking the first point of each line
            distance = Point(ray.coords[0]).distance(Point(point))
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        return LineString([ray.coords[0], closest_point])
    else:
        # If it's another type, return the original ray
        return ray  

def plot_data(data, camera_fov, camera_range, num_rays):
    fig, ax = plt.subplots()

    # Plotting obstacles
    for obstacle in data['obstacles']:
        coords = [(point['x'], point['y']) for point in obstacle['coordinates']]
        poly = Polygon(coords)
        patch = patches.Polygon(list(poly.exterior.coords), color='gray', alpha=0.5)
        ax.add_patch(patch)

    # Plotting cameras and their FOV
    for camera in data['cameras']:
        camera_pos = (float(camera['x']), float(camera['y']))
        direction = float(camera['direction'])
        print("Camera position and direction:", camera_pos, direction)  # Debug print

        # Calculate FOV polygon for each camera
        obstacles = [Polygon([(float(point['x']), float(point['y'])) for point in obstacle['coordinates']]) for obstacle in data['obstacles']]
        fov_polygon = calculate_coverage(camera_pos, direction, camera_fov, camera_range, num_rays, obstacles)

        if isinstance(fov_polygon, Polygon):
            fov_patch = patches.Polygon(list(fov_polygon.exterior.coords), color='green', alpha=0.3, closed=True)
            ax.add_patch(fov_patch)

        ax.add_patch(patches.Circle(camera_pos, radius=2, color='blue'))

    ax.set_xlim(-300, 1025)  # Adjust as per your coordinate range
    ax.set_ylim(-300, 1750)
    ax.set_aspect('equal', 'box')
    plt.show()

# Load data from JSON
def load_data_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

data = load_data_from_json('C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\camera_obstacle_data.json')


# Plot the data with FOV
plot_data(data, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS)