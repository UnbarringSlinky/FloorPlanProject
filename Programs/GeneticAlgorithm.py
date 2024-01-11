import random
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import MultiPoint, Point, LineString, Polygon, MultiPolygon, box
from math import exp
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from roboflow import Roboflow
from deap import base, creator, tools, algorithms

# Global variable for obstacles
obstacles = []

# Initializer function for the pool
def init_worker(obst):
    global obstacles
    obstacles = obst

# Problem Constants
NUM_CAMERAS = 7
CAMERA_FOV = 108  # Field of View in degrees
CAMERA_RANGE = 200  # Maximum distance camera can "see"
NUM_RAYS = 100
MAX_X = 700  # Maximum x-coordinate
MAX_Y = 600  # Maximum y-coordinate
orientation_step = 10  # This is the angular step in degrees. Adjust as needed for granularity.
orientations = np.arange(0, 360, orientation_step)
orientations = np.radians(orientations)

# Genetic Algorithm Constants
POPULATION_SIZE = 100
P_CROSSOVER = 0.7  # probability for crossover
P_MUTATION = 0.4   # probability for mutating an individual
MAX_GENERATIONS = 100
NUM_ELITES = 2

# Function for calculating a polygon that represents a single cameras view
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

def calculate_building_center(obstacles):
    # Collect all exterior coordinates from the obstacle polygons
    all_coords = []
    for obstacle in obstacles:
        # Assuming each obstacle is a Polygon object
        all_coords.extend(list(obstacle.exterior.coords))

    # Create a polygon from all the coordinates and find its centroid
    if all_coords:
        building_polygon = Polygon(all_coords)
        return building_polygon.centroid
    else:
        return None

def calculate_radial_coverage_score(center, coverage_polygons, num_rays=720):
    angle_step = 360 / num_rays
    coverage_score = 0
    gap_penalty = 0
    current_gap_size = 0

    for angle in np.arange(0, 360, angle_step):
        angle_rad = np.radians(angle)
        # Create a long line from center in the current direction
        line_end = (center.x + np.cos(angle_rad) * 1000, center.y + np.sin(angle_rad) * 1000)
        line = LineString([center, line_end])

        # Check if line intersects any camera's FOV polygon
        if any(line.intersects(polygon) for polygon in coverage_polygons):
            coverage_score += 1
            if current_gap_size > 0:
                # Apply penalty for the gap that just ended
                gap_penalty += current_gap_size ** 1.75  # Squared penalty
                current_gap_size = 0  # Reset gap size
        else:
            current_gap_size += 1  # Increase gap size

    # Check for a gap at the end wrapping around to the start
    if current_gap_size > 0:
        gap_penalty += current_gap_size ** 1.75

    return (coverage_score / num_rays) - gap_penalty

# Define function to penalize gaps between coverage_polygons
def calculate_polygon_gap_penalty(coverage_polygons):
    gap_penalty = 0

    # Ensure all polygons are valid and non-degenerate
    valid_polygons = [poly.buffer(0) for poly in coverage_polygons if poly.is_valid and not poly.is_empty]

    # Iterate through all unique pairs of valid coverage polygons
    for i, poly1 in enumerate(valid_polygons):
        for poly2 in valid_polygons[i+1:]:
            # Calculate the shortest distance between the two polygons
            gap_distance = poly1.distance(poly2)
            # Apply a penalty based on the gap distance
            gap_penalty += gap_distance ** 1.4

    return gap_penalty

# Define Fitness Function
def evaluate(individual):
    # Debugging print pid
    # print(f"Evaluating individual on process with PID: {os.getpid()}")

    # Max distance to wall (For penalties)
    max_distance_from_wall = 20
    wall_distance_penalty = 0
    wall_penalty_value = 500

    # Calculate unique coverage area
    coverage_polygons = []
    for camera in individual:
        camera_pos = (camera[0], camera[1])  # Extract x and y coordinates as a tuple
        direction = camera[2]               # Extract direction
        coverage_polygon = calculate_coverage(camera_pos, direction, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles)
        coverage_polygons.append(coverage_polygon)

    # Create a MultiPolygon object from individual coverage areas
    total_coverage = MultiPolygon([p.buffer(0) for p in coverage_polygons if p.is_valid])
    unique_coverage_area = 0
    if isinstance(total_coverage, Polygon):
        unique_coverage_area = total_coverage.area
    else:
        for polygon in total_coverage.geoms:
            adjusted_area = polygon.area - sum(polygon.intersection(other).area for other in total_coverage.geoms if other != polygon)
            unique_coverage_area += adjusted_area
            
    # Proximity to walls
    proximity_to_walls_score = sum(1 / (Point(camera[0], camera[1]).distance(nearest_wall(obstacles, camera[0], camera[1])) + 1) for camera in individual)

    # Distance from other cameras (increased weight)
    distance_from_cameras_score = sum(np.hypot(camera[0] - other_camera[0], camera[1] - other_camera[1]) for i, camera in enumerate(individual) for other_camera in individual[i+1:])

    # Calculate building center
    center = calculate_building_center(obstacles)

    # Calculate radial coverage score
    radial_coverage_score = calculate_radial_coverage_score(center, coverage_polygons)

    # Calculate coverage_polygon_gap penalty
    coverage_polygon_gap_penalty = calculate_polygon_gap_penalty(coverage_polygons)

    # Diversity metric - Encourages spreading across the plot
    diversity_score = calculate_diversity_score(individual, MAX_X, MAX_Y)

    # Penalize for clustering
    #clustering_penalty = 0
    #for i, camera in enumerate(individual):
    #    for other_camera in individual[i+1:]:
    #        distance = np.hypot(camera[0] - other_camera[0], camera[1] - other_camera[1])
    #        if distance < 90:  # Cameras are too close to each other
    #            clustering_penalty += 1  # Increase penalty for each close pair
    
    # Penalize for Facing similar directions and Facing towards eachother.
    #facing_each_other_penalty = 0
    #angular_tolerance = np.radians(30)  # Tolerance in radians, adjust as needed
    #for i, cam1 in enumerate(individual):
    #    for cam2 in individual[i+1:]:
    #        if is_direct_line_clear(cam1, cam2, obstacles):
    #            angle_cam1_to_cam2 = np.degrees(np.arctan2(cam2[1] - cam1[1], cam2[0] - cam1[0]))
    #            angle_cam2_to_cam1 = (angle_cam1_to_cam2 + 180) % 360
    #            if abs(angle_cam1_to_cam2 - cam1[2]) < angular_tolerance and \
    #               abs(angle_cam2_to_cam1 - cam2[2]) < angular_tolerance:
    #                facing_each_other_penalty += 1

    # calculate corner distance and distance to wall penalty
    corners = get_corners_of_walls(obstacles)
    proximity_to_corners_score = 0
    for camera in individual:
        camera_pos = Point(camera[0], camera[1])
        nearest_corner_distance = min(camera_pos.distance(corner) for corner in corners)
        proximity_to_corners_score += 1 / (nearest_corner_distance + 1)  # Inverse of distance to reward closeness
        distance = nearest_wall_distance(obstacles, camera[0], camera[1])

        # Apply a penalty if camera is farther than the max distance from wall
        if distance > max_distance_from_wall:
            wall_distance_penalty += wall_penalty_value

    # Penalize for placing cameras inside of walls
    inside_wall_penalty = 0
    illegal_placement_penalty_value = 100000  # Large penalty value to prevent any cameras from being inside walls.

    for camera in individual:
        camera_pos = Point(camera[0], camera[1])
        if any(camera_pos.within(wall) for wall in obstacles):
            inside_wall_penalty += illegal_placement_penalty_value
    # Combine scores with weights
    coverage_weight = .7 # Weight for promoting maximum coverage (Most Important)
    radial_coverage_weight = 1.2 
    wall_proximity_weight = .9 # Weight for promoting being closer to walls
    camera_distance_weight = .5 # Weight for promoting more distance from other cameras
    diversity_weight = .8  # Weight for diversity -- Promotes Cameras that are farther from any other camera
    #line_of_sight_weight = .7 # Facing Eachother Penalty Weight
    #clustering_weight = .5
    corner_proximity_weight = 0.75

    # Calculate total score
    total_score = (coverage_weight * unique_coverage_area +
                   wall_proximity_weight * proximity_to_walls_score +
                   camera_distance_weight * distance_from_cameras_score +
                   radial_coverage_weight * radial_coverage_score +
                   corner_proximity_weight * proximity_to_corners_score +
                   diversity_weight * diversity_score -
                   wall_distance_penalty - 
                   coverage_polygon_gap_penalty -
                   #line_of_sight_weight * facing_each_other_penalty - 
                   inside_wall_penalty) 
    return total_score,
# Calculate Coords of Corners
def get_corners_of_walls(obstacles):
    corners = []
    for obstacle in obstacles:
        # Assuming obstacles are polygons, extract their corner points
        xs, ys = obstacle.exterior.coords.xy
        corners.extend([Point(x, y) for x, y in zip(xs, ys)])
    return corners

# Calculate diversity in space
def calculate_diversity_score(individual, max_x, max_y):
    x_coords = [camera[0] for camera in individual]
    y_coords = [camera[1] for camera in individual]
    std_dev_x = np.std(x_coords)
    std_dev_y = np.std(y_coords)

    # Normalize the standard deviations by the maximum dimensions
    normalized_std_dev_x = std_dev_x / max_x
    normalized_std_dev_y = std_dev_y / max_y

    # Combine and scale the normalized standard deviations
    diversity_score = (normalized_std_dev_x + normalized_std_dev_y) * 0.5  # Scale factor
    return diversity_score

# Find and calculate the closest point in an obstacle and find the distance from a specified x,y position
def nearest_wall_distance(obstacles, x, y):
    point = Point(x, y)
    nearest_wall = min(obstacles, key=lambda wall: point.distance(wall))
    return point.distance(nearest_wall)

# Function to find nearest wall from a camera position
def nearest_wall(obstacles, x, y):
    return min(obstacles, key=lambda wall: Point(x, y).distance(wall))

# Function to find if two cameras are clear of any walls between eachother
def is_direct_line_clear(cam1, cam2, obstacles):
    line = LineString([(cam1[0], cam1[1]), (cam2[0], cam2[1])])
    return not any(line.intersects(obstacle) for obstacle in obstacles)

def custom_mutate(individual, indpb, sigma):
    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutate x coordinate
            individual[i] = (individual[i][0] + random.gauss(0, sigma),
                             # Mutate y coordinate
                             individual[i][1] + random.gauss(0, sigma),
                             # Mutate angle
                             (individual[i][2] + random.gauss(0, sigma)) % 360)
    return individual,

#Define Functions for calculating camera placement, Camera coverage, camera adding, wall adding and greedy functions
def trim_ray_at_obstacle(ray, obstacles):
    intersections = [ray.intersection(obstacle) for obstacle in obstacles if ray.intersects(obstacle)]
    if intersections:
        closest_intersection = min(intersections, key=lambda x: Point(ray.coords[0]).distance(x))
        return LineString([ray.coords[0], closest_intersection.coords[0]])
    return ray

# Todo Add on click events for adding cameras and moving them.
# def on_click(event):

# Todo add function for manual optimization of cameras


# Create a rectangle as a polygon from the given dimensions
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

# Function to check if camera is placed within an obstacle
def is_camera_inside_obstacle(camera_pos, obstacles):
    camera_point = Point(camera_pos)
    return any(camera_point.within(obstacle) for obstacle in obstacles)

# Algorithm for immigration camera direction (Greedy approach)
def find_best_orientation_for_camera(camera_pos, coverage_polygons):
    best_direction = 0  # Default direction
    best_coverage_polygon = None
    max_covered_area = 0

    for direction in orientations:
        coverage_polygon = calculate_coverage(camera_pos, direction, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles)
        covered_area = coverage_polygon.area - sum([coverage_polygon.intersection(other).area for other in coverage_polygons if other != coverage_polygon])

        if covered_area > max_covered_area:
            max_covered_area = covered_area
            best_direction = direction
            best_coverage_polygon = coverage_polygon

    return best_direction, best_coverage_polygon

# Function to create immigrants such that all genes will produce valid camera placements (Outside of obstacles)
def generate_immigrant():
    coverage_polygons = []  # Initialize an empty list to store coverage polygons
    immigrant_attributes = []

    for _ in range(NUM_CAMERAS):
        while True:
            x = random.uniform(0, MAX_X)
            y = random.uniform(0, MAX_Y)
            if not is_camera_inside_obstacle((x, y), obstacles):
                break

        # Find the best direction for this camera position
        best_direction, best_coverage_polygon = find_best_orientation_for_camera((x, y), coverage_polygons)
        immigrant_attributes.append((x, y, best_direction))

        # Update coverage_polygons with the newly calculated coverage polygon
        if best_coverage_polygon is not None:
            coverage_polygons.append(best_coverage_polygon)

    # Convert the list of attributes into a DEAP individual
    return creator.Individual(immigrant_attributes)

# Function for handling multiprocessing
def generate_single_immigrant(_):
    return generate_immigrant()

# Function pointer for Workers
def worker_evaluate(task_queue, result_queue):
    while True:
        individual = task_queue.get()
        if individual is None:  # Sentinel value to stop
            break
        fitness = evaluate(individual)
        result_queue.put((individual, fitness))

def main():
    global obstacles
    #JSON data from Object detection model.
    #import roboflow api and model
    rf = Roboflow(api_key="ToE2nfuMFjtdpjUPUL8f")
    project = rf.workspace().project("wall-floorplan")
    model = project.version(1).model

    # infer on a local image and populate obstacles
    data = model.predict("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\Data\\fp1.jpg", confidence=20, overlap=50).json()
    obstacles = [create_wall(item['x'], item['y'], item['width'], item['height'], MAX_Y) for item in data['predictions']]

    # instantiate pool for multiprocessing
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(obstacles,))

    # Set up DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Sigma is the magnitude of genetic mutations Larger values increase the magnitude of a change
    SIGMA = 1

    toolbox = base.Toolbox()

    # Attribute generator (These are all of the genes of each individual)
    toolbox.register("attr_float_x", random.uniform, 0, MAX_X)
    toolbox.register("attr_float_y", random.uniform, 0, MAX_Y)
    toolbox.register("attr_angle", random.uniform, 0, 360)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: (toolbox.attr_float_x(), toolbox.attr_float_y(), toolbox.attr_angle()),
                     NUM_CAMERAS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.2, sigma=SIGMA)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)
    for gen in range(MAX_GENERATIONS):
        # Evaluate all individuals in the population
        fitnesses = pool.map(evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select elites
        elites = tools.selBest(population, NUM_ELITES)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - NUM_ELITES)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = pool.map(evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Introduce Immigrants and evaluate their fitness
        # Generate immigrants in parallel
        immigrant_args = [None] * 10  # List of 10 'None' values, each will be passed to generate_single_immigrant
        immigrants = pool.map(generate_single_immigrant, immigrant_args)

        immigrant_fitnesses = pool.map(evaluate, immigrants)
        for ind, fit in zip(immigrants, immigrant_fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population with the offspring and elites
        population[:] = elites + offspring[:-10] + immigrants

        # Update the hall of fame with the new population
        hall_of_fame.update(population)

        # Optional: Print/log current generation statistics
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        maximum = max(fits)
    
        print(f"Generation: {gen}")
        print(f"  Min Fitness: {min(fits)}")
        print(f"  Max Fitness: {maximum}")
        print(f"  Avg Fitness: {mean}")
        print(f"  Num Elites: {NUM_ELITES}")
    pool.close()
    pool.join()
    # Final selection of the best individual
    best_ind = hall_of_fame[0]
    print("Best Individual = ", best_ind)
    print("Best Fitness = ", best_ind.fitness.values[0])

    # Assuming best_ind contains the best camera positions and orientations
    placed_cameras = []
    total_coverage = []

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-100, 925)
    ax.set_ylim(-100, 850)

    # Plot obstacles
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        ax.add_patch(MplPolygon(np.column_stack((x, y)), color='red', alpha=0.5))

    # Plot cameras and their coverage
    for camera in best_ind:
        print("Camera:", camera) 
        # Extract camera position and direction
        camera_pos = (camera[0], camera[1])  # Extract x and y coordinates as a tuple
        direction = camera[2]               # Extract direction

        coverage_area = calculate_coverage(camera_pos, direction, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles)
        x, y = coverage_area.exterior.xy
        ax.add_patch(MplPolygon(np.column_stack((x, y)), color='blue', alpha=0.3, edgecolor='blue'))

        # Plot camera position
        ax.scatter(*camera_pos, color='green', s=30, zorder=5)

    plt.show()

    # Format camera data into a list to be saved for future use
    camera_data = [{'x': camera[0], 'y': camera[1], 'direction': camera[2]} for camera in best_ind]

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

if __name__ == "__main__":
    main()