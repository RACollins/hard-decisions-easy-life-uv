from manim import *

class TestScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))