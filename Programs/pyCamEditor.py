from shapely.geometry import MultiPoint, Point, Polygon, LineString
import shapely.ops as ops
import numpy as np
import json
import pygame
import math

# Constants for the window size and offset
WINDOW_WIDTH = 1225
WINDOW_HEIGHT = 1150
OFFSET_X = 300
OFFSET_Y = 300

# Functions
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
    if intersections:
        closest_intersection = min(intersections, key=lambda x: Point(ray.coords[0]).distance(x))
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    return ray

# Load data from the file
def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Save data to the file
def save_data(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))

# Load camera and obstacle data
data = load_data('camera_obstacle_data.json')
cameras = data['cameras']
obstacles = [Polygon(obstacle['coordinates']) for obstacle in data['obstacles']]
NUM_RAYS = 100
# Draw obstacles cameras and Fov
def draw(cameras, obstacles, fov_angle, max_distance):
    screen.fill((255, 255, 255))  # Clear the screen with white

    # Draw obstacles directly without translation
    for obstacle in obstacles:
        pygame.draw.polygon(screen, (0, 0, 0), [(int(x), int(y)) for x, y in obstacle.exterior.coords])

    # Draw cameras and their FOV
    for camera in cameras:
        # Camera's position is used directly
        camera_screen_pos = (int(camera['x']), int(camera['y']))
        pygame.draw.circle(screen, (0, 0, 255), camera_screen_pos, 10)  # Draw camera

        # Calculate and draw camera's FOV
        direction = camera['direction']
        fov_polygon = calculate_coverage((camera['x'], camera['y']), direction, fov_angle, max_distance, NUM_RAYS, obstacles)
        if isinstance(fov_polygon, Polygon):
            pygame.draw.polygon(screen, (0, 255, 0, 50), [(int(x), int(y)) for x, y in fov_polygon.exterior.coords])

    pygame.display.flip()  # Update the full display surface to the screen





# Function to find the camera closest to a point
def find_closest_camera(cameras, point):
    closest_camera = None
    min_distance = float('inf')
    for camera in cameras:
        distance = math.hypot(camera['x'] - point[0], camera['y'] - point[1])
        if distance < min_distance:
            min_distance = distance
            closest_camera = camera
    return closest_camera

running = True
selected_camera = None
fov_angle = 108  # Field of view in degrees
max_distance = 200  # Maximum view distance
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos

            # Use Pygame's mouse position directly
            if event.button == 1:  # Left click to select
                selected_camera = find_closest_camera(cameras, (click_x, click_y))
            elif event.button == 3 and selected_camera:  # Right click to place
                selected_camera = None

        elif event.type == pygame.MOUSEMOTION:
            if selected_camera and pygame.mouse.get_pressed()[0]:
                # Update the camera's position with the mouse position
                selected_camera['x'], selected_camera['y'] = event.pos

        elif event.type == pygame.KEYDOWN:
            if selected_camera:
                if event.key == pygame.K_UP:
                    selected_camera['direction'] += 5
                elif event.key == pygame.K_DOWN:
                    selected_camera['direction'] -= 5

    draw(cameras, obstacles, fov_angle, max_distance)



# Format camera data into a list to be saved for future use
camera_data = [{'x': camera['x'], 'y': camera['y'], 'direction': camera['direction']} for camera in cameras]

# Format Obstacle data into a serializable format
obstacle_data = []
for obstacle in obstacles:
    coords = list(obstacle.exterior.coords)
    obstacle_data.append({'coordinates': coords})

# Combine into a single data structure
data_to_save = {'cameras': camera_data, 'obstacles': obstacle_data}

# Save to a json file
with open('camera_obstacle_data.json', 'w') as file:
    json.dump(data_to_save, file, indent=4)

pygame.quit()
