import random
import json
import copy
import numpy as np
import math
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
NUM_CAMERAS = 6
CAMERA_FOV = 114  # Field of View in degrees
CAMERA_RANGE = 500  # Maximum distance camera can "see"
NUM_RAYS = 100
MAX_X = 1200  # Maximum x-coordinate
MAX_Y = 800  # Maximum y-coordinate
orientation_step = 10  # This is the angular step in degrees. Adjust as needed for granularity.
orientations = np.arange(0, 360, orientation_step)
orientations = np.radians(orientations)

# Genetic Algorithm Constants
POPULATION_SIZE = 200
P_CROSSOVER = 0.7  # probability for crossover
P_MUTATION = 0.4   # probability for mutating an individual
MAX_GENERATIONS = 25
NUM_ELITES = 1

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
    all_coords = []

    for obstacle in obstacles:
        # For Polygon objects, use exterior coordinates
        if isinstance(obstacle, Polygon):
            all_coords.extend(list(obstacle.exterior.coords))
        # For LineString objects, use its coordinates directly
        elif isinstance(obstacle, LineString):
            all_coords.extend(list(obstacle.coords))
        # Add additional checks here if there are other types of geometries

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

# Function to determine if the camera is within the building. Can be used as a way to keep cameras inside or outside
def is_camera_inside_building(center, camera, obstacles):
    # Extract x and y coordinates from the camera Point object
    camera_pos = (camera.x, camera.y)

    # Create a line (ray) from center through camera position and beyond
    direction = (camera_pos[0] - center.x, camera_pos[1] - center.y)
    ray_end = (camera_pos[0] + direction[0] * 1000, camera_pos[1] + direction[1] * 1000)
    ray = LineString([center, ray_end])

    # Find the farthest obstacle along the ray
    farthest_obstacle_distance = 0
    for obstacle in obstacles:
        if ray.intersects(obstacle):
            intersection = ray.intersection(obstacle)
            distance = center.distance(intersection)
            if distance > farthest_obstacle_distance:
                farthest_obstacle_distance = distance

    # Check if camera is between center and the farthest obstacle
    camera_distance = center.distance(Point(camera_pos))
    return camera_distance <= farthest_obstacle_distance

# Function used to calculate if there is any gaps (Used to give big bonus for full coverage)
def calculate_total_gap_rays(center, coverage_polygons, num_rays=2880):
    angle_step = 360 / num_rays
    total_gap_rays = 0

    for angle in np.arange(0, 360, angle_step):
        angle_rad = np.radians(angle)
        # Create a long line from the center in the current direction
        line_end = (center.x + np.cos(angle_rad) * 1000, center.y + np.sin(angle_rad) * 1000)
        line = LineString([center, line_end])

        # Check if the line intersects any camera's FOV polygon
        if not any(line.intersects(polygon) for polygon in coverage_polygons):
            total_gap_rays += 1  # Increment gap count if no intersection

    return total_gap_rays

# Define function to calculate total overlap between coverage polygons
def calculate_total_overlap(coverage_polygons):
    total_overlap = 0

    # Ensure all polygons are valid and non-degenerate
    valid_polygons = [poly.buffer(0) for poly in coverage_polygons if poly.is_valid and not poly.is_empty]

    # Iterate through all unique pairs of valid coverage polygons
    for i, poly1 in enumerate(valid_polygons):
        for poly2 in valid_polygons[i+1:]:
            # Check if polygons intersect
            if poly1.intersects(poly2):
                # Calculate the area of intersection (overlap) between the two polygons
                overlap_area = poly1.intersection(poly2).area
                # Add to the total overlap
                total_overlap += overlap_area ** 1.5

    return total_overlap

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
    max_distance_from_wall = 50
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
        
    # Calculate building center
    center = calculate_building_center(obstacles)

    # Calculate radial coverage score
    radial_coverage_score = calculate_radial_coverage_score(center, coverage_polygons)

    # Calculate coverage_polygon_gap penalty
    coverage_polygon_gap_penalty = calculate_polygon_gap_penalty(coverage_polygons)


    coverage = calculate_total_gap_rays(center, coverage_polygons)
    full_coverage_bonus = 0

    
    # calculate corner distance and distance to wall penalty
    illegal_placement_penalty_value = 10000000  # Large penalty value to prevent any cameras from being inside walls or too far from walls.
    is_illegal = False
    inside_building_penalty = 0
    facing_center_penalty = 0
    corners = get_corners_of_walls(obstacles)
    proximity_to_corners_score = 0
    for camera in individual:
        is_facing_center = is_camera_facing_center(camera, center)
        camera_pos = Point(camera[0], camera[1])
        if is_camera_inside_building(center, camera_pos, obstacles):
            inside_building_penalty += illegal_placement_penalty_value
            is_illegal = True
        if is_facing_center:
            facing_center_penalty += illegal_placement_penalty_value
            is_illegal = True
        nearest_corner_distance = min(camera_pos.distance(Point(corner)) for corner in corners)  # Convert each corner to Point
        proximity_to_corners_score += 1 / (nearest_corner_distance + 1)  # Inverse of distance to reward closeness
        distance = nearest_wall_distance(obstacles, camera[0], camera[1])

    

        # Apply a penalty if camera is farther than the max distance from wall
        if distance > max_distance_from_wall:
            wall_distance_penalty += illegal_placement_penalty_value
    
    # Penalize for overlapping areas ## TODO Try promoting touching FOVs
    overlap_penalty = calculate_total_overlap(obstacles)
    # Penalize for placing cameras inside of walls
    inside_wall_penalty = 0
    for camera in individual:
        camera_pos = Point(camera[0], camera[1])
        if any(camera_pos.within(wall) for wall in obstacles):
            inside_wall_penalty += illegal_placement_penalty_value
            is_illegal = True
    # Combine scores with weights
    coverage_weight = 1 # Weight for promoting maximum coverage (Most Important)
    radial_coverage_weight = 1 

    if coverage == 0 and not is_illegal:
        full_coverage_bonus = 500000000
        print("Full Coverage Achieved!")

    # Calculate total score
    total_score = (coverage_weight * unique_coverage_area +
                   full_coverage_bonus +
                   radial_coverage_weight * radial_coverage_score -
                   wall_distance_penalty - 
                   coverage_polygon_gap_penalty -
                   inside_wall_penalty - 
                   inside_building_penalty - 
                   facing_center_penalty) 
    return (total_score,)

# Calculate Coords of Corners
def get_corners_of_walls(obstacles):
    corners = []

    for obstacle in obstacles:
        # For Polygon objects, use the exterior's coordinates
        if isinstance(obstacle, Polygon):
            corners.extend(list(obstacle.exterior.coords))
        # For LineString objects, use its coordinates directly
        elif isinstance(obstacle, LineString):
            corners.extend(list(obstacle.coords))
        # Add additional checks here if there are other types of geometries

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

        # Handle multi-part geometries
        if "Multi" in closest_intersection.geom_type:
            if isinstance(closest_intersection, MultiPoint):
                # Handle MultiPoint differently
                closest_part = min(closest_intersection.geoms, key=lambda part: Point(ray.coords[0]).distance(part))
            else:
                # For other Multi geometries like MultiPolygon or MultiLineString
                closest_part = min(closest_intersection, key=lambda part: Point(ray.coords[0]).distance(part))
            
            return LineString([ray.coords[0], closest_part.coords[0]])
        else:
            return LineString([ray.coords[0], closest_intersection.coords[0]])

    return ray

# Todo Add on click events for adding cameras and moving them.
# def on_click(event):

# Todo add function for manual optimization of cameras

def create_shape_from_points(points, max_y):
    # Flip the y-coordinate for each point
    flipped_points = [(point['x'], max_y - point['y']) for point in points]

    # Create a LineString for diagonal or complex shapes
    return LineString(flipped_points)

def create_shape(item, max_y):
    if 'points' in item and isinstance(item['points'], list):
        points = item['points']
        if len(points) > 2:
            # Use create_polygon for more than 2 points
            center_x, center_y, width, height = item['x'], item['y'], item['width'], item['height']
            return create_polygon(center_x, center_y, width, height, max_y)
        else:
            # Use create_linestring for 2 or fewer points
            return create_linestring(points, max_y)
    return None

def create_linestring(points, max_y):
    print("Created LineString")
    return LineString([(point['x'], max_y - point['y']) for point in points])

def create_polygon(center_x, center_y, width, height, max_y):
    half_width = width / 2
    half_height = height / 2

    # Calculate the top left corner based on the center point
    top_left_x = center_x - half_width
    top_left_y = center_y - half_height

    # Flip the y-coordinate for each corner
    y_flipped_top_left = max_y - top_left_y
    y_flipped_bottom_left = max_y - (top_left_y + height)

    # Create the polygon using the flipped y-coordinates
    return Polygon([(top_left_x, y_flipped_top_left), 
                    (top_left_x + width, y_flipped_top_left), 
                    (top_left_x + width, y_flipped_bottom_left), 
                    (top_left_x, y_flipped_bottom_left)])

def is_camera_facing_center(camera, center):
    # Extract camera position and direction
    camera_x, camera_y, camera_direction = camera

    # Calculate angle between camera and center
    delta_x = center.x - camera_x
    delta_y = center.y - camera_y
    angle_to_center = math.degrees(math.atan2(delta_y, delta_x))

    # Normalize angles to 0-360 degrees
    angle_to_center = angle_to_center % 360
    camera_direction = camera_direction % 360

    # Calculate the difference in angles
    angle_diff = abs(camera_direction - angle_to_center)

    # Check if the camera is approximately facing the center
    # Allowing a tolerance in degrees (e.g., 45 degrees on either side of the center direction)
    tolerance = 45
    return angle_diff <= tolerance or angle_diff >= (360 - tolerance)


def calculate_largest_gap_direction(center, coverage_polygons, num_rays=720):
    angle_step = 360 / num_rays
    gaps = []
    current_gap_start = None

    for angle in np.arange(0, 360, angle_step):
        angle_rad = np.radians(angle)
        line_end = (center.x + np.cos(angle_rad) * 1000, center.y + np.sin(angle_rad) * 1000)
        line = LineString([center, line_end])

        if any(line.intersects(polygon) for polygon in coverage_polygons):
            if current_gap_start is not None:
                # End of the current gap
                gaps.append((current_gap_start, angle))
                current_gap_start = None
        else:
            if current_gap_start is None:
                # Start of a new gap
                current_gap_start = angle

    # Check if a gap extends to the end of the range
    if current_gap_start is not None:
        gaps.append((current_gap_start, 360))

    # Find the largest gap
    if not gaps:
        return None  # No gaps found
    largest_gap = max(gaps, key=lambda gap: gap[1] - gap[0])
    # Calculate the middle direction of the largest gap
    largest_gap_direction = (largest_gap[0] + largest_gap[1]) / 2

    return largest_gap_direction

def evaluate_individual_with_camera_removed(individual, camera_index_to_remove):
    modified_individual = [camera for i, camera in enumerate(individual) if i != camera_index_to_remove]
    calculated_fitness = evaluate(modified_individual)
    print(f"Evaluated without least impactful camera: {calculated_fitness}")
    return calculated_fitness

def find_least_impactful_camera(individual):
    fitnesses_without_camera = []
    for camera_index in range(len(individual)):
        fitness_without_camera = evaluate_individual_with_camera_removed(individual, camera_index)
        fitnesses_without_camera.append((camera_index, fitness_without_camera))
    # Find the camera whose removal has the least impact on fitness
    
    least_impactful_camera_index, _ = max(fitnesses_without_camera, key=lambda x: x[1])
    print(f"    Chosen Eval with camera index: {least_impactful_camera_index}")
    return least_impactful_camera_index

def is_line_of_sight_clear(start_point, end_point, obstacles):
    line_of_sight = LineString([start_point, end_point])
    return not any(line_of_sight.intersects(obstacle) for obstacle in obstacles)

def relocate_camera_to_radial_gap(camera, center, largest_radial_gap_angle, minimum_distance=90, step=10, max_distance=1000):
    # Calculate new position of the camera
    angle_rad = np.radians(largest_radial_gap_angle)
    distance = minimum_distance

    while distance <= max_distance:
        new_x = center.x + np.cos(angle_rad) * distance
        new_y = center.y + np.sin(angle_rad) * distance

        max_x = center.x + np.cos(angle_rad) * max_distance
        max_y = center.y + np.sin(angle_rad) * max_distance

        # Check if the new position is outside of all obstacles and has a clear line of sight
        new_position = Point(new_x, new_y)
        max_position = Point(max_x, max_y)

        if not is_camera_inside_obstacle(new_position, obstacles) and is_line_of_sight_clear(new_position, max_position, obstacles) and not is_camera_inside_building(center, Point(new_x, new_y), obstacles):
            # Update camera position
            # Update camera direction (facing away from the center)
            new_direction = calculate_direction(center.x, center.y, new_x, new_y)
            return (new_x, new_y, new_direction) #Todo may not actually be needed to do np.degrees

        # Increase distance for the next iteration
        distance += step

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

def calculate_direction(center_x, center_y, target_x, target_y):
    """
    Calculate the direction from a center point to a target point.

    Args:
    - center_x, center_y: coordinates of the center point
    - target_x, target_y: coordinates of the target point

    Returns:
    - angle in degrees from the north
    """
    dx = target_x - center_x
    dy = target_y - center_y

    # Calculate the angle in radians and then convert to degrees
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)

    # Adjust the angle so that it is measured from the north
    # and ensure it is within the range [0, 360)
    north_adjusted_angle = (90 - angle_degrees) % 360  #if north include 90 - angle_degrees

    return north_adjusted_angle

# Function to create immigrants such that all genes will produce valid camera placements (Outside of obstacles)
def generate_immigrant():
    coverage_polygons = []  # Initialize an empty list to store coverage polygons
    immigrant_attributes = []
    # Find the center of the building
    center = calculate_building_center(obstacles)
    for _ in range(NUM_CAMERAS):
        while True:
            x = random.uniform(0, MAX_X)
            y = random.uniform(0, MAX_Y)
            camera = Point(x,y)
            if is_camera_inside_building(center, camera, obstacles):
                continue
            if not is_camera_inside_obstacle(camera, obstacles):
                break

        # Find the best direction for this camera position
        best_direction, best_coverage_polygon = find_best_orientation_for_camera((x, y), coverage_polygons)
        if is_camera_facing_center((x,y,best_direction), center):
            best_direction = calculate_direction(center.x, center.y, x, y)
            best_coverage_polygon = calculate_coverage((x, y), best_direction, CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles)
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
    rf = Roboflow(api_key="x")
    project = rf.workspace().project("floorplan-tracer")
    model = project.version(4).model

    # infer on a local image and populate obstacles
    data = model.predict("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\Data\\fp2.jpg", confidence=1).json()
    obstacles = []
    # Iterate over each prediction and create shapes
    for item in data['predictions']:
        # Check if the item's class is not 'background'
        if item.get('class') != 'background':
            if 'points' in item and isinstance(item['points'], list):
                shape = create_shape_from_points(item['points'], MAX_Y)
                obstacles.append(shape)


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
        
        # Identify the elite individual
        elite_individual = copy.deepcopy(tools.selBest(population, 1)[0])

        # Find the center of the building
        center = calculate_building_center(obstacles)

        # Calculate coverage polygons for the elite individual
        elite_coverage_polygons = [calculate_coverage((camera[0], camera[1]), camera[2], CAMERA_FOV, CAMERA_RANGE, NUM_RAYS, obstacles) for camera in elite_individual]

        # Identify the least impactful camera in the elite individual
        least_impactful_camera_index = find_least_impactful_camera(elite_individual)

        # Find the direction of the largest radial gap
        largest_gap_direction = calculate_largest_gap_direction(center, elite_coverage_polygons)

        # Relocate the least impactful camera to the largest radial gap
        if largest_gap_direction is not None:
            elite_individual[least_impactful_camera_index] = relocate_camera_to_radial_gap(elite_individual[least_impactful_camera_index], center, largest_gap_direction)

        #Replace the 11th individual from the end in the population with this modified elite individual
        replace_index = max(len(population) - 11, 0)
        population[replace_index] = creator.Individual(elite_individual)


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
        if isinstance(obstacle, LineString):
            x, y = obstacle.xy
            ax.plot(x, y, color='red', linewidth=2)  # Plot line
        elif isinstance(obstacle, Polygon):
            x, y = obstacle.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='red', ec='none')  # Fill polygon

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
        if isinstance(obstacle, Polygon):
            coords = list(obstacle.exterior.coords)
        elif isinstance(obstacle, LineString):
            coords = list(obstacle.coords)
        else:
            continue  # Skip non-Polygon and non-LineString objects

        # Serialize the coordinates
        serialized_coords = [{'x': coord[0], 'y': coord[1]} for coord in coords]
        obstacle_data.append({'coordinates': serialized_coords})

    # Combine into a single data structure
    data_to_save = {'cameras': camera_data, 'obstacles': obstacle_data}

    # Save to a JSON file
    with open('camera_obstacle_data.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)

if __name__ == "__main__":
    main()
