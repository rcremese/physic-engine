import taichi as ti
import taichi.math as tm
from typing import Any
import numpy as np


GRAVITY = tm.vec2([0, -9.81])
WIDTH, HEIGHT = 1080, 720


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
    _scale_vector = np.array([WIDTH, HEIGHT])
    obs_size = 100

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
        self._old_pos = ti.Vector.field(n=2, dtype=ti.f32, shape=(n_balls))
        # Obstacles
        self.obstacle = Box(
            tm.vec2((self._scale_vector - self.obs_size) / 2),
            tm.vec2((self._scale_vector + self.obs_size) / 2),
        )

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

            self._old_pos[n] = self.balls[n].pos

    def run(self):
        self.reset()
        gui = ti.GUI("CanonBall2D", res=(WIDTH, HEIGHT))
        while gui.running:
            self.update()
            pos = self.balls.pos.to_numpy() / self._scale_vector
            # Balls
            gui.circles(
                pos,
                radius=10,
                color=0xED553B,
            )
            # Velocity
            gui.arrows(
                orig=pos,
                direction=self.balls.vel.to_numpy() / self._scale_vector,
                color=0xEEEEF0,
                radius=1,
            )
            # Obstacle
            top_left = self.obstacle.min.to_numpy() / self._scale_vector
            bottomright = self.obstacle.max.to_numpy() / self._scale_vector

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
            self.box_collision(n)
            if self.obstacle.ball_collision(self.balls[n]):
                self.obstacle_collision(n)
            # Update the old position
            self._old_pos[n] = self.balls[n].pos

    @ti.func
    def box_collision(self, n: int):
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

    @ti.func
    def obstacle_collision(self, n: int):
        # Check collision along x axis
        if (
            self._old_pos[n].x < self.obstacle.min.x
            and self.balls[n].pos.x >= self.obstacle.min.x
        ):
            self.balls[n].pos.x = self.obstacle.min.x
            self.balls[n].vel.x *= -1
        elif (
            self._old_pos[n].x > self.obstacle.max.x
            and self.balls[n].pos.x <= self.obstacle.max.x
        ):
            self.balls[n].pos.x = self.obstacle.max.x
            self.balls[n].vel.x *= -1
        # Check collision along y axis
        if (
            self._old_pos[n].y < self.obstacle.min.y
            and self.balls[n].pos.y >= self.obstacle.min.y
        ):
            self.balls[n].pos.y = self.obstacle.min.y
            self.balls[n].vel.y *= -1
        elif (
            self._old_pos[n].y > self.obstacle.max.y
            and self.balls[n].pos.y <= self.obstacle.max.y
        ):
            self.balls[n].pos.y = self.obstacle.max.y
            self.balls[n].vel.y *= -1


if __name__ == "__main__":
    ti.init(ti.gpu)
    ball = Ball(radius=1)
    box = Box.field(shape=())
    fps = 60
    canonball2d = CanonBall2D(np.array([1, HEIGHT - 1]), dt=1 / fps, substeps=1)
    canonball2d.run()
