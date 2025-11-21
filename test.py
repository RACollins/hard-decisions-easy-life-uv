from manim import Scene, Create, Square


class TestScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
