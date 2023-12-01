import taichi as ti
import taichi.math as tm
import gymnasium as gym
import numpy as np

@ti.dataclass
class State:
    """State of a point mass
    """
    pos : tm.vec2
    vel : tm.vec2
    mass : float

@ti.kernel
def map_to_pixel(positions : ti.template(), resolution : tm.vec2) -> None:
    """Map the positions to pixel coordinates
    """
    pixels = ti.zero(positions)
    for i in positions:
        pixels[i] = positions[i] / resolution
    return pixels

@ti.data_oriented
class LarvaSimulation:
    """Simulation of a larva using XPBD
    """
    def __init__(self, nb_parts : int = 1, mass : float = 1.) -> None:
        assert nb_parts >= 1, "nb_seg must be >= 1"
        self.nb_parts = int(nb_parts)
        self._nb_particles = 2*(self.nb_parts + 2)
        
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=self._nb_particles)
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=self._nb_particles)
        assert mass > 0, "mass must be > 0"
        self.mass = float(mass)

    def render(self) -> None:
        """Render the larva simulation
        """
        res= tm.ivec2(640, 360)
        window = ti.ui.Window(name='XPBD Larva', res = tuple(res), fps_limit=60, pos = (150, 150))
        canvas = window.get_canvas()
        while window.running:
            self.update()
            # pixels = map_to_pixel(self.positions, window.res)
            canvas.circles(centers=map_to_pixel(self.positions, res), radius=0.1, color=(0.5, 0.5, 0.5))
            window.show()

    def _init_positions(self, offset : float = 0.) -> None:
        """Initialize the positions of the particles
        """
        assert offset >= 0, "offset must be >= 0"
        if self.nb_parts == 1:
            self.positions[0] = tm.vec2(-2., 0.5 + offset) # Tail
            self.positions[1] = tm.vec2(-0.5, 0 + offset)
            self.positions[2] = tm.vec2(-0.5, 1 + offset)
            self.positions[3] = tm.vec2(0.5, 0 + offset)
            self.positions[4] = tm.vec2(0.5, 1 + offset)
            self.positions[5] = tm.vec2(2., 0.5 + offset) # Head
        else:
            raise NotImplementedError("Not implemented for nb_parts > 1")

    def update(self) -> None:
        """Update the simulation
        """
        pass

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    sim = LarvaSimulation(1)
    sim.render()