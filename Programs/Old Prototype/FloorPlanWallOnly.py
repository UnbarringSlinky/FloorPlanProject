import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, box
from roboflow import Roboflow
rf = Roboflow(api_key="x")
project = rf.workspace().project("floor_plan_wall_only-ebquk")
model = project.version(4).model

# infer on a local image
data = model.predict("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\fp1.jpg", confidence=10, overlap=90).json()

# visualize your prediction
model.predict("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\fp1.jpg", confidence=10, overlap=99).save("C:\\Users\\ryanl\\OneDrive\\Desktop\\FloorPlanProject\\pred1.jpg")

#Function to create a rectangle wall from data
def create_wall(x, y, width, height):
    return box(x, y, x + width, y + height)

#Process each wall and create a list of rectangles
walls = []
for item in data['predictions']:
    wall = create_wall(item['x'], item['y'], item['width'], item['height'])
    walls.append(wall)

#Set up plot
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

#plot each wall
for wall in walls:
    minx, miny, maxx, maxy = wall.bounds
    ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=None, edgecolor='blue'))

# Set limits and save the figure
ax.set_xlim([0, max(wall.bounds[2] for wall in walls)]) # Adjust according to your data
ax.set_ylim([0, max(wall.bounds[3] for wall in walls)]) # Adjust according to your data
plt.gca().invert_yaxis() # Invert y-axis to match the floorplan orientation
plt.savefig('floorplan.png', dpi=300)
