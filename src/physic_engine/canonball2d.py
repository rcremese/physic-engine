import taichi as ti
import taichi.math as tm
from typing import Any
import numpy as np


GRAVITY = tm.vec2([0, -9.81])
WIDTH, HEIGHT = 640, 480


@ti.dataclass
class Ball:
    pos: tm.vec2
    vel: tm.vec2
    inv_mass: float
    radius: float


class Box:
    width: int
    height: int

    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height


@ti.data_oriented
class CanonBall2D:
    def __init__(
        self,
        init_pos: np.ndarray,
        n_balls: int = 1,
        dt: float = 1,
        substeps: int = 1,
    ) -> None:
        assert init_pos.shape == (2,), "init_pos must be a 2D vector"
        self.init_pos = init_pos
        assert n_balls > 0, "n_balls must be positive"
        self.n_balls = n_balls

        self.balls = Ball.field(shape=(n_balls))

        self.box = Box(WIDTH, HEIGHT)
        assert dt > 0, "dt must be positive"
        self.dt = dt
        assert substeps > 0, "substeps must be positive"
        self.substeps = substeps

    @ti.kernel
    def reset(self):
        for n in self.balls:
            self.balls[n].pos = tm.vec2(self.init_pos)
            self.balls[n].vel = tm.vec2([10, 0])
            self.balls[n].inv_mass = 10.0
            self.balls[n].radius = 1

    def run(self):
        self.reset()
        gui = ti.GUI("CanonBall2D", res=(WIDTH, HEIGHT))
        while gui.running:
            self.update()
            pos = self.balls.pos.to_numpy() / np.array([WIDTH, HEIGHT])
            gui.circles(
                pos,
                radius=10,
                color=0xED553B,
            )
            gui.arrows(
                orig=pos,
                direction=self.balls.vel.to_numpy() / np.array([WIDTH, HEIGHT]),
                color=0xEEEEF0,
                radius=1,
            )
            gui.show()

    def update(self):
        for _ in range(self.substeps):
            self.step(self.dt / self.substeps)
        self.render()

    def render(self):
        pass

    @ti.kernel
    def step(self, dt: float):
        # ti.atomic_add(self.balls[n].vel, GRAVITY * self.balls[n].inv_mass * dt)
        # ti.atomic_add(self.balls[n].pos, self.balls[n].vel * dt)
        for n in self.balls:
            self.balls[n].vel += GRAVITY * self.balls[n].inv_mass * dt
            self.balls[n].pos += self.balls[n].vel * dt
            ## Collision detection

            self.collision(n)

    @ti.func
    def collision(self, n: int):
        # Left
        if self.balls[n].pos[0] < self.balls[n].radius:
            self.balls[n].pos[0] = self.balls[n].radius
            self.balls[n].vel[0] *= -1
        # Right
        if self.balls[n].pos[0] > self.box.width - self.balls[n].radius:
            self.balls[n].pos[0] = self.box.width - self.balls[n].radius
            self.balls[n].vel[0] *= -1
        # Bottom
        if self.balls[n].pos[1] < self.balls[n].radius:
            self.balls[n].pos[1] = self.balls[n].radius
            self.balls[n].vel[1] *= -1
        # Top
        if self.balls[n].pos[1] > self.box.height - self.balls[n].radius:
            self.balls[n].pos[1] = self.box.height - self.balls[n].radius
            self.balls[n].vel[1] *= -1


if __name__ == "__main__":
    ti.init(ti.gpu)
    canonball2d = CanonBall2D(np.array([WIDTH / 2, HEIGHT / 2]), dt=1 / 60, substeps=1)
    canonball2d.run()
