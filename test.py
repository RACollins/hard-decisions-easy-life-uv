from manim import Scene, Create, Transform, Square, Circle


class TestScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.play(Transform(square, circle))
