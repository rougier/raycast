# raycast.py - Raycast maze simulator
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Maze simulator using the raycast Digital Differential Analyzer (DDA) algorithm.
"""
import math
import numpy as np

def raycast(origin, direction, maze):
    """Compute the end coordinates of a ray that is cast from given
    origin and direction, until it reaches a wall or exit the
    maze. The physical size of the maze is normalized to fit the unit
    square originating at (0,0). The method is called Digital
    Differential Analyzer (DDA)

    Parameters
    ----------
   
    origin: tuple

      Origin of the ray that must be inside the maze

    direction: float

      Direction of the ray (radians)

    maze: ndarray

      2D array describing maze occupancy where value > 0 means
      occupîed and 0 means empty.

    Returns
    -------
  
    end_position: tuple

      End point of the ray in normalized coordinates or None if the
      ray exits the maze without hitting a wall
    
    end_cell: tuple
    
      End point of the ray in grid coordinates or None if the ray
      exits the maze without hitting a wall


    face : tuple

      Face that was hit: (1,0), (-1,0), (0,1) or (0, -1)
    
    steps: int

      Number of steps before hitting a wall (debug)

    """
 
    # Cell size from normalized maze dimension
    cell_size = 1/max(maze.shape)

    # Grid position
    x, y = origin[0] / cell_size, origin[1] / cell_size

    # Ray direction
    dx, dy = math.cos(direction), math.sin(direction)

    # Raycasting loop
    steps = 0
    cell_x_prev, cell_y_prev = x, y
    
    while True:
        steps += 1
        
        # Determine the cell the ray is currently in
        cell_x, cell_y = math.floor(x), math.floor(y)

        # Check if the ray exits the maze without hitting a wall
        if cell_x < 0 or cell_y < 0 or cell_x >= maze.shape[1] or cell_y >=  maze.shape[0]:
            return None, None, None, steps 

        # Check for wall collision
        if maze[cell_y, cell_x] > 0:
            face = cell_x_prev - cell_x, cell_y_prev - cell_y
            return np.array([x*cell_size, y*cell_size]), (cell_y, cell_x), face, steps

        cell_x_prev, cell_y_prev = cell_x, cell_y
        
        # Calculate next grid crossing
        epsilon = 1e-10
        if dx > epsilon:    tx = (cell_x + 1 - x) / dx
        elif dx < -epsilon: tx = (cell_x - x) / dx
        else:               tx = float('inf')

        if dy > epsilon:    ty = (cell_y + 1 - y) / dy
        elif dy < -epsilon: ty = (cell_y - y) / dy
        else:               ty = float('inf')

        # Ensure t_min is strictly positive to move the ray forward
        t_min = max(min(tx, ty), cell_size/10)  # Small epsilon ensures progress

        # Move ray to next cell boundary, talking vertical/horizontal cases into account
        epsilon = 1e-15
        if abs(dx) < epsilon:
            y += np.sign(dy)
        elif abs(dy) < epsilon:
            x += np.sign(dx)
        else:
            x += dx * t_min
            y += dy * t_min

class Camera:
    """ A camera based on raycasting """
    
    def __init__(self, fov = 60, resolution = 128):
        """Build a new camera with a fov (field of view) and a given resolution"""
        
        self.fov = fov
        self.resolution = resolution
        self.depths = np.zeros(resolution)
        self.rays = np.zeros((resolution,2,2))
        self.cells = np.zeros((resolution,2), dtype=int)
        self.framebuffer = np.zeros((resolution,resolution,3), dtype=np.ubyte)

    def render(self, position, direction, maze, cmap, outline=True, lighting=True):

        """Update the framebuffer with the current view and records
        the rays that have been used (for debug)"""
        
        # Clear framebuffer by coloring sky (top half) and ground (bottom half)
        n = self.resolution
        self.framebuffer[n//2:, :] = cmap[-2] * np.linspace(0.75, 1.00, n//2).reshape(n//2,1,1)
        self.framebuffer[:n//2, :] = cmap[-1] * np.linspace(1.00, 0.65, n//2).reshape(n//2,1,1)

        # Cast each ray and write them in the framebuffer
        angles = direction + np.radians(np.linspace(+self.fov/2,-self.fov/2, n, endpoint=True))

        cell_prev = None
        face_prev = None
        start = position
        for i, angle in enumerate(angles):
            end, cell, face, steps = raycast(start, angle, maze)
            d = math.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)

            self.rays[i] = start, end
            self.depths[i] = d
            self.cells[i] = cell
            
            # We should use a fovy instead of hardcoding height
            height = 0.09/(d*math.cos(direction - angle))

            ymin = max (math.floor((0.5 - height/2)*n), 0)
            ymax = min (math.floor((0.5 + height/2)*n), n)
            depth = abs(d) - abs(d) % (1/n)
            self.framebuffer[ymin:ymax,i] = cmap[maze[cell]]*(1 - depth)

            # Basic lighting
            if lighting and face == (-1,0):
                self.framebuffer[ymin:ymax,i] = 0.75 * self.framebuffer[ymin:ymax,i]
            
            # Wall outline
            if outline:
                if i > 0:
                    if np.any(self.cells[i-1] != self.cells[i]):
                        self.framebuffer[ymin:ymax,i] = self.framebuffer[ymin:ymax,i]*0.75
                    if np.all(self.cells[i-1] == self.cells[i]) and face != face_prev:
                        self.framebuffer[ymin:ymax,i] = self.framebuffer[ymin:ymax,i]*0.75
                self.framebuffer[ymax-1:ymax,i] = self.framebuffer[ymax-1,i]*0.75 
                self.framebuffer[ymin:ymin+1,i] = self.framebuffer[ymin,  i]*0.75

            face_prev = face

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,4,4,0,0,3,3,0,1],
        [1,0,4,0,0,0,0,3,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,6,0,0,0,0,5,0,1],
        [1,0,6,6,0,0,5,5,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]])
    
    cmap = {
        -2 : np.array([100, 100, 255]), # sky
        -1 : np.array([255, 255, 255]), # ground
         0 : np.array([255, 255, 255]), # empty
         1 : np.array([200, 200, 200]), # wall (light)
         2 : np.array([100, 100, 100]), # wall (dark)
         3 : np.array([255, 255,   0]), # wall (yellow)
         4 : np.array([  0,   0, 255]), # wall (blue)
         5 : np.array([255,   0,   0]), # wall (red)
         6 : np.array([  0, 255,   0])  # wall (green)
    }
    maze_rgb = [cmap[i] for i in maze.ravel()]
    maze_rgb = np.array(maze_rgb).reshape(maze.shape[0], maze.shape[1], 3)
    
    radius = 0.05
    position, direction = (0.5, 0.5),  np.radians(0)
    camera = Camera(fov = 60, resolution = 512)
    
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
    ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
    ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
    ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()
    ax1.imshow(maze_rgb, interpolation="nearest", origin="lower",
               extent = [0.0, maze.shape[1]/max(maze.shape),
                         0.0, maze.shape[0]/max(maze.shape)])
    ax1.add_artist(Circle(position, radius, zorder=50, facecolor="white", edgecolor="black"))
    rays = ax1.add_collection(LineCollection(camera.rays, color="C1", linewidth=0.5, zorder=30))
    framebuffer = ax2.imshow(camera.framebuffer, interpolation="nearest",
                             origin="lower", extent = [0.0, 1.0, 0.0, 1.0])

    def update(frame=0):
        global position, direction
        direction += math.radians(0.5)
        camera.render(position, direction, maze, cmap, outline=True)
        rays.set_segments(camera.rays)
        framebuffer.set_data(camera.framebuffer)

    # update(), fig.savefig("raycast.png")        
    ani = FuncAnimation(fig, update, frames=360, interval=1, repeat=True)
    # ani.save(filename="raycast.mp4", writer="ffmpeg", fps=30)
    plt.show()
