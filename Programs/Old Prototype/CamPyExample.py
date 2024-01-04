import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon

def trim_ray_at_obstacle(ray, obstacles):
    """Trims the ray at the first obstacle intersection."""
    intersections = [ray.intersection(obstacle) for obstacle in obstacles if ray.intersects(obstacle)]
    if intersections:
        closest_intersection = min(intersections, key=lambda x: Point(ray.coords[0]).distance(x))
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    return ray

# Camera setup
camera_pos = (25, 30) # Center of the area
fov_angle = np.radians(90) # 90 degrees FOV, facing downwards
max_distance = 20 # Max distance camera can "see"
num_rays = 100 # Number of rays to simulate the FOV

# Obstacles setup
obstacles = [Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
             Polygon([(30, 20), (40, 20), (40, 30), (30, 30)])]

# Ray casting within FOV to construct the visible area
fov_points = [camera_pos]
for angle in np.linspace(-fov_angle / 2, fov_angle / 2, num_rays):
    ray_end = (camera_pos[0] + np.cos(angle) * max_distance, camera_pos[1] + np.sin(angle) * max_distance)
    ray = LineString([camera_pos, ray_end])
    trimmed_ray = trim_ray_at_obstacle(ray, obstacles)
    fov_points.append(trimmed_ray.coords[-1])

fov_polygon = Polygon(fov_points) # Visible area polygon

# Plotting
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)

# Plot obstacles
for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='red', ec='none')

# Plot visible area
x, y = fov_polygon.exterior.xy
ax.fill(x, y, alpha=0.5, fc='blue', ec='none')

# Plot camera
plt.scatter(*camera_pos, color='green')

plt.show()
