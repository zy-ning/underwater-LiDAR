import numpy as np
from manim import *


class UBlock(VGroup):
    """A helper to create a visual block representing a tensor/layer."""

    def __init__(
        self, width, height, color, label_text="", show_neurons=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.rect = Rectangle(
            width=width, height=height, color=color, fill_opacity=0.2, stroke_width=2
        )
        self.add(self.rect)

        if label_text:
            label = Text(label_text, font_size=16, color=WHITE, font="Times New Roman").move_to(
                self.rect.get_center()
            )
            self.add(label)

        # Neurons for Dropout visualization
        self.neurons = VGroup()
        if show_neurons:
            rows = max(1, int(height * 4))
            cols = max(1, int(width * 4))
            for i in range(rows):
                for j in range(cols):
                    dot = Dot(radius=0.03, color=color)
                    # Position relative to rect
                    x = -width / 2 + (width / (cols + 1)) * (j + 1)
                    y = -height / 2 + (height / (rows + 1)) * (i + 1)
                    dot.move_to(self.rect.get_center() + np.array([x, y, 0]))
                    self.neurons.add(dot)
            self.add(self.neurons)

    def apply_dropout(self, p=0.3):
        """Randomly dim neurons to simulate dropout."""
        for neuron in self.neurons:
            if np.random.random() < p:
                neuron.set_opacity(0.1)
            else:
                neuron.set_opacity(1.0)

    def reset_dropout(self):
        self.neurons.set_opacity(1.0)


class MCDUNetVisualizer(Scene):
    def construct(self):
        # ==========================================
        # 1. SETUP TITLES AND LAYOUT
        # ==========================================
        title = Text("SPAD Peak Detection: Monte Carlo Dropout U-Net", font_size=30, font="Times New Roman")
        title.to_edge(UP)
        self.play(Write(title))

        # ==========================================
        # 2. DRAW THE U-NET ARCHITECTURE
        # ==========================================
        # Configuration for visual balance
        level_colors = [BLUE, TEAL, GREEN, YELLOW]

        # Create blocks
        # Encoder
        e1 = UBlock(1.5, 3.0, level_colors[0], "In", show_neurons=True)
        e2 = UBlock(1.2, 2.0, level_colors[1], "", show_neurons=True)
        e3 = UBlock(1.0, 1.2, level_colors[2], "", show_neurons=True)
        bottleneck = UBlock(2.0, 0.8, level_colors[3], "Bottleneck", show_neurons=True)

        # Decoder
        d3 = UBlock(1.0, 1.2, level_colors[2], "", show_neurons=True)
        d2 = UBlock(1.2, 2.0, level_colors[1], "", show_neurons=True)
        d1 = UBlock(1.5, 3.0, level_colors[0], "Out", show_neurons=True)

        # Positioning (The "U" shape)
        # Shift down slightly to make room for graphs later
        center_y = -0.5

        e1.move_to(LEFT * 4 + UP * 1.5 + DOWN * 0.5)
        e2.next_to(e1, RIGHT + DOWN, buff=0.5)
        e3.next_to(e2, RIGHT + DOWN, buff=0.5)
        bottleneck.next_to(e3, DOWN, buff=0.5).set_x(0)  # Center bottom

        # Mirror for decoder
        d3.next_to(bottleneck, UP, buff=0.5).set_x(1)
        d3.align_to(e3, UP)
        d2.next_to(d3, RIGHT + UP, buff=0.5)
        d2.align_to(e2, UP)
        d1.next_to(d2, RIGHT + UP, buff=0.5)
        d1.align_to(e1, UP)

        # Adjust X positions symmetrically
        e3.set_x(-1)
        d3.set_x(1)
        e2.set_x(-2.5)
        d2.set_x(2.5)
        e1.set_x(-4)
        d1.set_x(4)

        unet_group = VGroup(e1, e2, e3, bottleneck, d3, d2, d1)

        # Connections (Arrows)
        arrows = VGroup()
        # Down
        arrows.add(Arrow(e1.get_bottom(), e2.get_top(), buff=0.1, color=GREY))
        arrows.add(Arrow(e2.get_bottom(), e3.get_top(), buff=0.1, color=GREY))
        arrows.add(Arrow(e3.get_bottom(), bottleneck.get_top(), buff=0.1, color=GREY))
        # Up
        arrows.add(Arrow(bottleneck.get_top(), d3.get_bottom(), buff=0.1, color=GREY))
        arrows.add(Arrow(d3.get_top(), d2.get_bottom(), buff=0.1, color=GREY))
        arrows.add(Arrow(d2.get_top(), d1.get_bottom(), buff=0.1, color=GREY))

        # Skip Connections
        skips = VGroup()
        skips.add(
            DashedLine(e3.get_right(), d3.get_left(), color=WHITE, stroke_opacity=0.5)
        )
        skips.add(
            DashedLine(e2.get_right(), d2.get_left(), color=WHITE, stroke_opacity=0.5)
        )
        skips.add(
            DashedLine(e1.get_right(), d1.get_left(), color=WHITE, stroke_opacity=0.5)
        )

        skip_label = Text("Skip Connections", font_size=18, slant=ITALIC, font="Times New Roman").next_to(
            skips[1], UP, buff=0.1
        )

        self.play(FadeIn(unet_group), Create(arrows))
        self.play(Create(skips), Write(skip_label))
        self.wait(1)

        # ==========================================
        # 3. SETUP INPUT/OUTPUT GRAPHS
        # ==========================================

        # Scale down U-Net to fit graphs
        full_arch = VGroup(unet_group, arrows, skips, skip_label)
        self.play(full_arch.animate.scale(0.6).to_edge(DOWN, buff=0.5))

        # Axes
        ax_in = (
            Axes(
                x_range=[0, 10, 5],
                y_range=[0, 5, 5],
                x_length=3,
                y_length=2,
                axis_config={"include_numbers": False, "stroke_width": 1},
            )
            .to_edge(UP)
            .shift(LEFT * 5.0)
            .shift(DOWN * 1.0)
        )

        ax_out = (
            Axes(
                x_range=[0, 10, 5],
                y_range=[0, 1, 1],
                x_length=3,
                y_length=2,
                axis_config={"include_numbers": False, "stroke_width": 1},
            )
            .to_edge(UP)
            .shift(RIGHT * 5.0)
            .shift(DOWN * 1.0)
        )

        lab_in = Text("Noisy Input", font_size=20, font="Times New Roman").next_to(ax_in, DOWN)
        lab_out = Text("Prediction (Dist)", font_size=20, font="Times New Roman").next_to(ax_out, DOWN)

        # Create Noisy Input Curve
        def get_noisy_input(x):
            signal = 3 * np.exp(-((x - 4) ** 2) / 0.05)  # Peak at 4
            noise = np.random.normal(0, 1) + 0.5  # Ambient
            return max(0, signal + noise)

        input_curve = ax_in.plot(
            lambda x: 3 * np.exp(-((x - 4) ** 2) / 0.05)
            + 0.5
            + np.random.normal(0, 0.1),
            color=BLUE_A,
        )

        self.play(
            Create(ax_in),
            Create(ax_out),
            Write(lab_in),
            Write(lab_out),
            Create(input_curve),
        )

        # ==========================================
        # 4. ANIMATE MC DROPOUT LOOP
        # ==========================================

        mc_text = Text("Monte Carlo Sampling...", font_size=20, color=YELLOW, font="Times New Roman").move_to(
            full_arch.get_top() + UP * 0.5
        )
        self.play(FadeIn(mc_text))

        predictions = VGroup()
        num_samples = 10

        # Connector lines (Data flow)
        flow_in = Line(
            ax_in.get_bottom(),
            full_arch.get_left(),
            color=BLUE,
            stroke_width=2,
            stroke_opacity=0.5,
        )
        flow_out = Line(
            full_arch.get_right(),
            ax_out.get_bottom(),
            color=RED,
            stroke_width=2,
            stroke_opacity=0.5,
        )

        for i in range(num_samples):
            # 1. Update Dropout Masks (Visual Flicker)
            # We apply dropout to layers to show they change every pass
            anim_group = []
            for block in [e1, e2, e3, bottleneck, d3, d2, d1]:
                block.apply_dropout(p=0.4)
                # Quick flash effect to emphasize change
                anim_group.append(
                    block.rect.animate.set_stroke(opacity=1.0).set_stroke(opacity=0.2)
                )

            # 2. Simulate Signal Flow
            pulse = Dot(color=YELLOW).move_to(flow_in.get_start())

            # Animate this pass
            # We combine the dropout flicker with the pulse movement
            self.play(MoveAlongPath(pulse, flow_in), run_time=0.2, rate_func=linear)
            self.play(
                Wiggle(full_arch, scale_value=1.02, rotation_angle=0.01 * PI),
                run_time=0.2,
            )

            # 3. Generate a slightly different output curve
            # Shift peak slightly and change height to simulate uncertainty
            offset = np.random.normal(0, 0.2)
            amp_var = np.random.normal(0, 0.1)

            pred_func = lambda x: (1.0 + amp_var) * np.exp(
                -((x - (4 + offset)) ** 2) / 0.5
            )
            curve = ax_out.plot(
                pred_func, color=RED_A, stroke_width=2, stroke_opacity=0.5
            )
            predictions.add(curve)

            pulse_out = Dot(color=YELLOW).move_to(flow_out.get_start())
            self.play(MoveAlongPath(pulse_out, flow_out), FadeIn(curve), run_time=0.2)

            # Cleanup pulse
            self.remove(pulse, pulse_out)

        # ==========================================
        # 5. AGGREGATE RESULTS (MEAN + VAR)
        # ==========================================

        self.play(FadeOut(mc_text))
        agg_text = Text("Aggregation", font_size=24, color=GREEN, font="Times New Roman").move_to(
            mc_text.get_center()
        )
        self.play(Write(agg_text))

        # Calculate visual mean (approximate)
        mean_func = lambda x: 1.0 * np.exp(
            -((x - 4) ** 2) / 0.6
        )  # Slightly wider due to jitter
        mean_curve = ax_out.plot(mean_func, color=WHITE, stroke_width=4)

        # Calculate visual variance area
        # Upper bound
        upper_func = lambda x: 1.2 * np.exp(-((x - 4) ** 2) / 0.6) + 0.1
        lower_func = lambda x: 0.8 * np.exp(-((x - 4) ** 2) / 0.6) - 0.05

        area = ax_out.get_area(
            ax_out.plot(upper_func),
            bounded_graph=ax_out.plot(lower_func),
            color=BLUE,
            opacity=0.3,
        )

        # Transition: Fade faint curves into Mean + Area
        self.play(FadeOut(predictions), FadeIn(area), Create(mean_curve), run_time=2)

        # Labels for final plot
        lbl_mean = Text("Mean Prediction", color=WHITE, font_size=14, font="Times New Roman").next_to(
            mean_curve, UP, buff=0.1
        ).shift(RIGHT * 0.5)
        lbl_unc = (
            Text("Uncertainty", color=BLUE, font_size=14, font="Times New Roman")
            .move_to(area.get_center())
            .shift(RIGHT * 0.6)
        )

        self.play(Write(lbl_mean), Write(lbl_unc))

        self.wait(2)
