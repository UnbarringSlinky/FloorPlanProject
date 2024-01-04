import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from matplotlib.patches import Rectangle
from roboflow import Roboflow

#import roboflow api and model
rf = Roboflow(api_key="ToE2nfuMFjtdpjUPUL8f")
project = rf.workspace().project("wall-floorplan")
model = project.version(1).model

# infer on a local image
data = model.predict("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\fp1.jpg", confidence=47, overlap=1).json()

#Define Functions for calculating camera placement, Camera coverage, camera adding, wall adding and greedy functions
def trim_ray_at_obstacle(ray, obstacles):
    intersections = [ray.intersection(obstacle) for obstacle in obstacles if ray.intersects(obstacle)]
    if intersections:
        closest_intersection = min(intersections, key=lambda x: Point(ray.coords[0]).distance(x))
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    return ray

def calculate_coverage(camera_pos, direction, fov_angle, max_distance, num_rays, obstacles):
    fov_points = [camera_pos]
    start_angle = direction - fov_angle / 2
    end_angle = direction + fov_angle / 2
    for angle in np.linspace(start_angle, end_angle, num_rays):
        ray_end = (camera_pos[0] + np.cos(angle) * max_distance, camera_pos[1] + np.sin(angle) * max_distance)
        ray = LineString([camera_pos, ray_end])
        trimmed_ray = trim_ray_at_obstacle(ray, obstacles)
        fov_points.append(trimmed_ray.coords[-1])
    return Polygon(fov_points)

def on_click(event):
    if event.inaxes != ax: return
    camera_pos = (event.xdata, event.ydata)
    best_direction, best_coverage = find_best_orientation_for_camera(camera_pos)
    placed_cameras.append((camera_pos, best_direction))
    total_coverage.append(best_coverage)
    draw_coverage_area(best_coverage)
    plt.scatter(*camera_pos, color='green')
    fig.canvas.draw()

def find_best_orientation_for_camera(camera_pos):
    best_direction = None
    best_coverage_polygon = None
    max_covered_area = 0

    for direction in orientations:
        coverage_polygon = calculate_coverage(camera_pos, direction, fov_angle, max_distance, num_rays, obstacles)
        covered_area = coverage_polygon.area - sum([coverage_polygon.intersection(cov).area for cov in total_coverage])

        if covered_area > max_covered_area:
            max_covered_area = covered_area
            best_direction = direction
            best_coverage_polygon = coverage_polygon

    return best_direction, best_coverage_polygon

def draw_coverage_area(coverage_polygon):
    if isinstance(coverage_polygon, Polygon):
        x, y = coverage_polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='blue', ec='none')

def create_wall(center_x, center_y, width, height, max_y):
    half_width = width / 2
    half_height = height / 2

    # Calculate the top left corner based on the center point
    top_left_x = center_x - half_width
    top_left_y = center_y - half_height  # Subtract half height to move up from the center

    # Flip the y-coordinate for the top left corner
    y_flipped_top_left = max_y - top_left_y

    # Create the polygon using the top left corner and width and height
    return Polygon([(top_left_x, y_flipped_top_left), 
                    (top_left_x + width, y_flipped_top_left), 
                    (top_left_x + width, y_flipped_top_left - height), 
                    (top_left_x, y_flipped_top_left - height)])
# Import JSON data here and convert it into obstacles (list of Polygons)
walls = []
for item in data['predictions']:
    wall = create_wall(item['x'], item['y'], item['width'], item['height'],800) #800 is max_y
    walls.append(wall)
obstacles = walls

fov_angle = np.radians(90)  # 90 degrees FOV
max_distance = 200
num_rays = 100
orientations = np.radians(np.arange(0, 360, 45))
placed_cameras = []
total_coverage = []

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-400, 800)
ax.set_ylim(-400, 800)

for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='red', ec='none')

cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
