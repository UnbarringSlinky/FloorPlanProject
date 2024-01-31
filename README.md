This project is a sophisticated simulation and optimization tool designed to tackle the complex challenge of strategic surveillance camera placement within a given environment. Utilizing advanced genetic algorithms, this tool iteratively searches for the most effective camera configurations that maximize coverage area while minimizing blind spots and redundancy. The core of the project revolves around the ingenious use of genetic algorithm principles to evolve camera positions, orientations, and settings over successive generations, aiming for an optimal surveillance setup.

The environment for the surveillance system is defined by a series of obstacles, such as walls and barriers, which cameras must work around to achieve comprehensive coverage. Users can interactively adjust the layout, adding, or modifying obstacles and camera parameters to see how these changes affect overall coverage. This dynamic interaction allows for rapid prototyping and testing of surveillance strategies in varied layouts, providing invaluable insights into the strengths and weaknesses of each configuration.

Underpinning the project is a robust mathematical model that calculates the field of view for each camera, taking into account obstacles, camera range, and angle of view. This model ensures that the genetic algorithm works with accurate simulations of real-world conditions. Additionally, the project includes features for visualizing the evolutionary process, offering a graphical representation of camera placements and their fields of view as they evolve. This visualization not only makes it easier to understand how the algorithm improves coverage over time but also provides a clear demonstration of the final surveillance setup's effectiveness.

Designed with flexibility in mind, this project applies to a wide range of scenarios, from securing small indoor spaces to developing comprehensive surveillance strategies for large complexes. It stands as a testament to the power of combining evolutionary computation with interactive design tools, offering a cutting-edge solution to one of the critical challenges in security planning.

*To Install Dependencies*
pip install shapely
pip install deap
pip install numpy
pip install matplotlib
pip install pygame
