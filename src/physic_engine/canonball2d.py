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

    def bounding_box(self):
        return Box(self.pos - self.radius, self.pos + self.radius)


@ti.dataclass
class Box:
    min: tm.vec2
    max: tm.vec2

    @ti.func
    def width(self):
        return self.max[0] - self.min[0]

    @ti.func
    def height(self):
        return self.max[1] - self.min[1]

    @ti.func
    def ball_collision(self, ball: Ball):
        closest_point = tm.clamp(ball.pos, self.min, self.max)
        dist = tm.length(closest_point - ball.pos)
        return dist <= ball.radius


@ti.func
def learp(x, y, t):
    return x * (1 - t) + y * t


@ti.data_oriented
class CanonBall2D:
    def __init__(
        self,
        init_pos: np.ndarray,
        n_balls: int = 1,
        n_obstacles: int = 1,
        dt: float = 1,
        substeps: int = 1,
    ) -> None:
        assert init_pos.shape == (2,), "init_pos must be a 2D vector"
        self.init_pos = init_pos
        assert n_balls > 0, "n_balls must be positive"
        self.n_balls = n_balls
        self.balls = Ball.field(shape=(n_balls))
        # Obstacles
        assert n_obstacles > 0, "n_obstacles must be positive"
        self.n_obstacles = n_obstacles
        self.obstacles = Box.field(shape=(n_obstacles))

        self.box = Box(tm.vec2(0), tm.vec2([WIDTH, HEIGHT]))
        assert dt > 0, "dt must be positive"
        self.dt = dt
        assert substeps > 0, "substeps must be positive"
        self.substeps = substeps

    @ti.kernel
    def reset(self):
        for n in self.balls:
            self.balls[n].pos = tm.vec2(self.init_pos)
            self.balls[n].vel = tm.vec2([100, 100])
            self.balls[n].inv_mass = 10.0
            self.balls[n].radius = 1

        for o in self.obstacles:
            a = tm.vec2([ti.random() * WIDTH, ti.random() * HEIGHT])
            b = tm.vec2([ti.random() * WIDTH, ti.random() * HEIGHT])
            self.obstacles[o] = Box(tm.min(a, b), tm.max(a, b))

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
            obstacles = self.obstacles.to_numpy()
            for o in range(self.n_obstacles):
                top_left = (
                    obstacles["min"][o, 0] / WIDTH,
                    obstacles["max"][o, 1] / HEIGHT,
                )
                bottomright = (
                    obstacles["max"][o, 0] / WIDTH,
                    obstacles["min"][o, 0] / HEIGHT,
                )

                gui.rect(
                    topleft=top_left,
                    bottomright=bottomright,
                    color=0x0000FF,
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
        if self.balls[n].pos[0] > self.box.width() - self.balls[n].radius:
            self.balls[n].pos[0] = self.box.width() - self.balls[n].radius
            self.balls[n].vel[0] *= -1
        # Bottom
        if self.balls[n].pos[1] < self.balls[n].radius:
            self.balls[n].pos[1] = self.balls[n].radius
            self.balls[n].vel[1] *= -1
        # Top
        if self.balls[n].pos[1] > self.box.height() - self.balls[n].radius:
            self.balls[n].pos[1] = self.box.height() - self.balls[n].radius
            self.balls[n].vel[1] *= -1


if __name__ == "__main__":
    ti.init(ti.gpu)
    ball = Ball(radius=1)
    box = Box.field(shape=())
    fps = 60
    canonball2d = CanonBall2D(
        np.array([WIDTH / 2, HEIGHT / 2]), dt=1 / fps, substeps=1, n_obstacles=1
    )
    canonball2d.run()
