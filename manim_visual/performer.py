import numpy as np
from manim import *


class PerformerArchitectureDetail(Scene):
    def construct(self):
        self.camera.background_color = "#000000" # Dark blue-grey background

        # ==========================================
        # SCENE 1: THE INPUT SEQUENCE (LiDAR Data)
        # ==========================================

        # 1. Draw the Signal
        title = Text("Input: Long LiDAR Sequence (L = 1600 bins)", font_size=32).to_edge(UP)

        axes = Axes(
            x_range=[0, 100], y_range=[0, 1],
            x_length=10, y_length=2,
            axis_config={"include_ticks": False}
        ).shift(UP * 2)

        # Noisy Signal Function
        def signal_func(x):
            decay = 0.4 * np.exp(-0.05 * x)
            peak = 0.5 * np.exp(-0.5 * (x - 30)**2)
            noise = np.random.normal(0, 0.05)
            return max(0, decay + peak + noise)

        signal_graph = axes.plot(lambda x: signal_func(x), color=BLUE_C)

        self.play(Write(title), Create(axes), Create(signal_graph))
        self.wait(1)

        # 2. Tokenization / Embedding
        # Transform signal into a Matrix representation (Sequence Length L x Dimension d)
        input_matrix = Rectangle(height=4, width=1.5, fill_color=BLUE_E, fill_opacity=0.8, stroke_color=WHITE)
        input_matrix.move_to(LEFT * 5)

        input_label = MathTex("X \in \mathbb{R}^{L \\times d}").next_to(input_matrix, UP)

        self.play(
            FadeOut(axes), FadeOut(signal_graph),
            ReplacementTransform(signal_graph.copy(), input_matrix),
            Write(input_label)
        )

        # ==========================================
        # SCENE 2: Q, K, V PROJECTION
        # ==========================================

        # Create Q, K, V matrices derived from Input
        q_mat = Rectangle(height=3, width=1, fill_color=BLUE, fill_opacity=0.6).move_to(LEFT * 3)
        k_mat = Rectangle(height=3, width=1, fill_color=YELLOW, fill_opacity=0.6).move_to(LEFT * 0)
        v_mat = Rectangle(height=3, width=1, fill_color=RED, fill_opacity=0.6).move_to(RIGHT * 3)

        q_lbl = MathTex("Q").next_to(q_mat, UP)
        k_lbl = MathTex("K").next_to(k_mat, UP)
        v_lbl = MathTex("V").next_to(v_mat, UP)

        self.play(
            ReplacementTransform(input_matrix.copy(), q_mat),
            ReplacementTransform(input_matrix.copy(), k_mat),
            ReplacementTransform(input_matrix.copy(), v_mat),
            Write(q_lbl), Write(k_lbl), Write(v_lbl),
            FadeOut(input_matrix), FadeOut(input_label)
        )

        # ==========================================
        # SCENE 3: THE KERNEL TRICK (FAVOR+)
        # ==========================================

        # Explanation Text
        mechanism_title = Text("Performer Mechanism: FAVOR+", font_size=28, color=YELLOW).next_to(title, DOWN)
        self.play(Write(mechanism_title))

        # Show the Random Feature Map phi(x)
        kernel_box_q = RoundedRectangle(height=1, width=2, color=PURPLE).move_to(q_mat.get_center())
        kernel_box_k = RoundedRectangle(height=1, width=2, color=PURPLE).move_to(k_mat.get_center())

        kernel_text = MathTex("\phi(\cdot)", color=PURPLE).move_to(kernel_box_k)

        self.play(Create(kernel_box_q), Create(kernel_box_k), Write(kernel_text))

        # Transform Q and K into Prime versions (Q', K')
        # Visually, they change texture or slightly size to show projection
        q_prime = q_mat.copy().set_fill(TEAL).set_stroke(PURPLE)
        k_prime = k_mat.copy().set_fill(ORANGE).set_stroke(PURPLE)

        q_prime_lbl = MathTex("Q'").next_to(q_prime, UP)
        k_prime_lbl = MathTex("K'").next_to(k_prime, UP)

        self.play(
            ReplacementTransform(q_mat, q_prime),
            ReplacementTransform(k_mat, k_prime),
            Transform(q_lbl, q_prime_lbl),
            Transform(k_lbl, k_prime_lbl),
            FadeOut(kernel_box_q), FadeOut(kernel_box_k), FadeOut(kernel_text)
        )

        # ==========================================
        # SCENE 4: LINEAR ATTENTION (Changing the order)
        # ==========================================

        # Equation at bottom
        equation_std = MathTex(r"Attn(Q, K, V) = \text{softmax}(Q K^T) V").to_edge(DOWN).shift(UP*1)
        equation_perf = MathTex(r"Attn(Q, K, V) \approx Q' (K'^T V)").to_edge(DOWN).shift(UP*1)

        complexity_bad = Text("O(L^2) - Too Slow", color=RED, font_size=24).next_to(equation_std, DOWN)
        complexity_good = Text("O(L) - Linear Speed", color=GREEN, font_size=24).next_to(equation_perf, DOWN)

        self.play(Write(equation_std), FadeIn(complexity_bad))
        self.wait(1)

        # Visualize the Matrix Multiplication Change
        # Standard: Q x K (High cost)
        # Performer: K x V first!

        self.play(
            ReplacementTransform(equation_std, equation_perf),
            ReplacementTransform(complexity_bad, complexity_good)
        )

        # Animate K' Transpose * V
        # Rotate K
        k_transposed = k_prime.copy().rotate(PI/2).next_to(v_mat, LEFT, buff=0.2)

        self.play(
            k_prime.animate.set_opacity(0.2), # Dim original
            v_mat.animate.move_to(RIGHT * 2),
            ReplacementTransform(k_prime.copy(), k_transposed)
        )

        # Merge K'T and V into a "Global Context" matrix
        # Crucially, this matrix is SMALL (dim x dim), not (Length x Length)
        global_context = Square(side_length=1.5, fill_color=GREEN, fill_opacity=0.8).move_to(RIGHT * 2)
        context_lbl = Text("Global Context", font_size=20).move_to(global_context)

        self.play(
            ReplacementTransform(k_transposed, global_context),
            ReplacementTransform(v_mat, global_context),
            Write(context_lbl)
        )

        # Now Multiply Q' by Context
        self.play(
            q_prime.animate.next_to(global_context, LEFT, buff=0.5)
        )

        # Result Matrix
        result_matrix = Rectangle(height=4, width=1.5, fill_color=GREEN_E, stroke_color=WHITE).move_to(RIGHT * 5)
        result_lbl = Text("Output", font_size=24).next_to(result_matrix, UP)

        # Arrows showing flow
        arrow = Arrow(start=global_context.get_right(), end=result_matrix.get_left())

        self.play(
            ReplacementTransform(global_context.copy(), result_matrix),
            ReplacementTransform(q_prime.copy(), result_matrix),
            FadeOut(v_lbl),
            Create(arrow),
            Write(result_lbl)
        )

        self.wait(2)

        # ==========================================
        # SCENE 5: FINAL CLEAN SIGNAL
        # ==========================================

        # Show the result transforming back to a clean plot
        clean_axes = Axes(
            x_range=[0, 100], y_range=[0, 1],
            x_length=8, y_length=3
        ).move_to(ORIGIN)

        def clean_func(x):
            # Just the peak, no backscatter, no noise
            return 0.8 * np.exp(-0.5 * (x - 30)**2)

        clean_graph = clean_axes.plot(lambda x: clean_func(x), color=GREEN)
        clean_lbl = Text("Target Extracted", color=GREEN).next_to(clean_axes, UP)

        final_group = VGroup(clean_axes, clean_graph, clean_lbl)

        self.play(
            FadeOut(q_prime), FadeOut(q_lbl), FadeOut(k_prime), FadeOut(k_lbl),
            FadeOut(global_context), FadeOut(context_lbl), FadeOut(result_matrix),
            FadeOut(result_lbl), FadeOut(arrow), FadeOut(equation_perf), FadeOut(complexity_good),
            FadeIn(final_group)
        )

        self.wait(3)
