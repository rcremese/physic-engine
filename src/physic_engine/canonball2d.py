from typing import Any
import warp
import warp.sim as wp_sim
import warp.sim.render as wp_render
import numpy as np

GRAVITY = warp.vec2f([0, -9.81])


class Ball:
    pos: warp.vec2
    vel: warp.vec2
    inv_mass: float
    radius: float

    def __init__(self, pos, vel=np.zeros(2), inv_mass=1, radius=1) -> None:
        self.pos = warp.vec2f(pos)
        self.vel = warp.vec2f(vel)
        assert inv_mass > 0, "inv_mass must be positive"
        self.inv_mass = inv_mass
        assert radius > 0, "radius must be positive"
        self.radius = radius


class Box:
    width: int
    height: int

    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height


class CanonBall2D:
    def __init__(
        self, init_pos: np.ndarray, dt: float = 0.1, substeps: int = 1
    ) -> None:
        self.ball = Ball(init_pos)
        self.box = Box(100, 100)
        assert dt > 0, "dt must be positive"
        self.dt = dt
        assert substeps > 0, "substeps must be positive"
        self.substeps = substeps

    def update(self):
        for _ in range(self.substeps):
            self.simulate(self.dt / self.substeps)
        self.render()

    def render(self):
        pass

    def simulate(self, dt: float = 0.1):
        self.ball.vel += GRAVITY * self.ball.inv_mass * dt
        self.ball.pos += self.ball.vel * dt
        ## Collision detection
        # Left
        if self.ball.pos[0] < self.ball.radius:
            self.ball.pos[0] = self.ball.radius
            self.ball.vel[0] *= -1
        # Right
        if self.ball.pos[0] > self.box.width - self.ball.radius:
            self.ball.pos[0] = self.box.width - self.ball.radius
            self.ball.vel[0] *= -1
        # Bottom
        if self.ball.pos[1] < self.ball.radius:
            self.ball.pos[1] = self.ball.radius
            self.ball.vel[1] *= -1
        # Top
        if self.ball.pos[1] > self.box.height - self.ball.radius:
            self.ball.pos[1] = self.box.height - self.ball.radius
            self.ball.vel[1] *= -1


if __name__ == "__main__":
    canonball2d = CanonBall2D(np.ones(2))
    canonball2d.update()
