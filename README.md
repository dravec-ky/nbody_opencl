
This project implements a n-body problem approximation technique as a microsolver node inside 
Houdini FX. It uses the Barnes-Hut Approximation and its parallel implementation to sort the 
particles in the simulation into a tree-like structure, which is then traversed by each particle to 
calculate approximate gravitational force acting on it. This is programmed using OpenCL and 
implemented into Houdini, adding parameters which offer artistic control over the simulation and 
better performance compared to the base n-body algorithm.
