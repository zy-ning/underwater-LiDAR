import numpy as np
from manim import *
from scipy.stats import norm


# --- Helper Functions for Fake Data ---
def get_gaussian(x_range, mu, sigma, amp=1.0):
    x = np.linspace(x_range[0], x_range[1], 200)
    y = amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return x, y

def get_noisy_gaussian(x_range, mu, sigma, noise_scale=0.1):
    x, y = get_gaussian(x_range, mu, sigma)
    noise = np.random.normal(0, noise_scale, len(x))
    y = np.clip(y + noise, 0, None)
    return x, y

class SharedConfig:
    """Shared visual configuration"""
    AXES_CONFIG = {
        "x_range": [0, 100, 10],
        "y_range": [0, 1.2, 0.5],
        "x_length": 6,
        "y_length": 3,
        "axis_config": {"include_tip": False},
        "tips": False,
    }

class Introduction(Scene):
    def construct(self):
        title = Text("LiDAR Temporal Aggregation", font_size=48).to_edge(UP)
        subtitle = Text("Improving Peak Detection over N Periods", font_size=32, color=GRAY).next_to(title, DOWN)

        methods = VGroup(
            Text("1. Pre-Average (Early Fusion)", font_size=28),
            Text("2. Post-Average (Late Fusion)", font_size=28),
            Text("3. Max-Voting (Robust Fusion)", font_size=28),
            Text("4. Heteroscedastic Kalman Filter (Uncertainty-Aware)", font_size=28, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)

        self.play(Write(title), FadeIn(subtitle))
        self.play(Write(methods))
        self.wait(2)

class PreAverage(Scene):
    def construct(self):
        title = Text("1. Pre-Average (Early Fusion)", font_size=36).to_edge(UP)
        self.add(title)

        # Create 3 small axes for inputs
        axes_group = VGroup()
        signals = VGroup()

        for i in range(3):
            ax = Axes(**SharedConfig.AXES_CONFIG).scale(0.5)
            # Noisy raw signals
            x_vals, y_vals = get_noisy_gaussian([0, 100], 50, 5, noise_scale=0.2)
            graph = ax.plot_line_graph(x_vals, y_vals, add_vertex_dots=False, line_color=BLUE_C, stroke_width=2)

            label = Text(f"Period {i+1}", font_size=20).next_to(ax, DOWN)
            group = VGroup(ax, graph, label)
            axes_group.add(group)
            signals.add(graph)

        axes_group.arrange(RIGHT, buff=0.5).shift(UP*0.8)

        self.play(FadeIn(axes_group))
        self.wait(0.5)

        # Create result axis
        res_ax = Axes(**SharedConfig.AXES_CONFIG).scale(0.8).move_to(DOWN * 2)
        res_label = Text("Averaged Raw Input (High SNR)", font_size=24, color=GREEN).next_to(res_ax, DOWN)

        # Clean signal
        x_clean, y_clean = get_gaussian([0, 100], 50, 5)
        res_graph = res_ax.plot_line_graph(x_clean, y_clean, add_vertex_dots=False, line_color=GREEN, stroke_width=3)

        arrow_group = VGroup()
        for group in axes_group:
            arrow = Arrow(start=group.get_bottom(), end=res_ax.get_top(), buff=0.1, color=GRAY)
            arrow_group.add(arrow)

        self.play(GrowFromCenter(arrow_group))
        self.play(TransformFromCopy(signals, res_graph), FadeIn(res_ax), Write(res_label))

        text_explanation = Text("Average raw histograms\nbefore prediction.", font_size=24).to_corner(UL)
        self.play(Write(text_explanation))
        self.wait(2)

class PostAverage(Scene):
    def construct(self):
        title = Text("2. Post-Average (Late Fusion)", font_size=36).to_edge(UP)
        self.add(title)

        # Visualization: 3 Model Predictions -> Mean Curve
        axes_group = VGroup()
        preds = VGroup()

        # Generate 3 slightly different predictions (simulating jitter)
        mus = [48, 52, 50]

        for i, mu in enumerate(mus):
            ax = Axes(**SharedConfig.AXES_CONFIG).scale(0.5)
            x_vals, y_vals = get_gaussian([0, 100], mu, 3) # Clean prediction curves
            graph = ax.plot_line_graph(x_vals, y_vals, add_vertex_dots=False, line_color=RED_C, stroke_width=2)
            label = Text(f"Pred {i+1}", font_size=20).next_to(ax, DOWN)
            group = VGroup(ax, graph, label)
            axes_group.add(group)
            preds.add(graph)

        axes_group.arrange(RIGHT, buff=0.5).shift(UP*0.5)
        self.play(FadeIn(axes_group))

        # Result Axis
        res_ax = Axes(**SharedConfig.AXES_CONFIG).scale(0.8).move_to(DOWN * 2)
        res_label = Text("Mean of Probabilities", font_size=24, color=PURPLE).next_to(res_ax, DOWN)

        # The average of the gaussians
        avg_mu = np.mean(mus)
        x_avg, y_avg = get_gaussian([0, 100], avg_mu, 4) # Slightly wider due to jitter
        res_graph = res_ax.plot_line_graph(x_avg, y_avg, add_vertex_dots=False, line_color=PURPLE, stroke_width=3)

        self.play(FadeIn(res_ax), Write(res_label))

        # Animate curves moving down and merging
        self.play(
            ReplacementTransform(preds[0].copy(), res_graph),
            ReplacementTransform(preds[1].copy(), res_graph),
            ReplacementTransform(preds[2].copy(), res_graph)
        )

        explanation = Text("Average model outputs.\nReduces variance.", font_size=24).to_corner(UL)
        self.play(Write(explanation))
        self.wait(2)

class MaxVote(Scene):
    def construct(self):
        title = Text("3. Max-Voting (Robust Fusion)", font_size=36).to_edge(UP)
        self.add(title)

        # 3 Predictions: 2 agree, 1 is an outlier
        mus = [50, 50, 80] # Outlier at 80
        axes_group = VGroup()
        peaks = VGroup()

        for i, mu in enumerate(mus):
            ax = Axes(**SharedConfig.AXES_CONFIG).scale(0.5)
            x_vals, y_vals = get_gaussian([0, 100], mu, 2)
            graph = ax.plot_line_graph(x_vals, y_vals, add_vertex_dots=False, line_color=ORANGE, stroke_width=2)

            # Draw a line at the peak
            peak_line = DashedLine(
                start=ax.c2p(mu, 0), end=ax.c2p(mu, 1.0), color=WHITE
            )
            peak_text = Text(f"Peak: {mu}", font_size=16).next_to(peak_line, UP, buff=0.1)

            group = VGroup(ax, graph, peak_line, peak_text)
            axes_group.add(group)
            peaks.add(peak_line)

        axes_group.arrange(RIGHT, buff=0.5).shift(UP*0.9)
        self.play(FadeIn(axes_group))

        # Voting Box
        box = Rectangle(height=2, width=6, color=WHITE).move_to(DOWN * 1.5)
        vote_title = Text("Vote Tally", font_size=24).next_to(box, UP)

        tally_text = VGroup(
            Text(f"Bin 50: 2 Votes", color=GREEN),
            Text(f"Bin 80: 1 Vote", color=RED)
        ).arrange(DOWN).move_to(box.get_center())

        self.play(Create(box), Write(vote_title))
        self.play(Write(tally_text))

        # Result
        final_res = Text("Result: Bin 50", font_size=32, color=GREEN, weight=BOLD).next_to(box, DOWN)
        self.play(TransformFromCopy(tally_text[0], final_res))

        explanation = Text("Mode estimation.\nIgnores outliers.", font_size=24).to_corner(UL)
        self.play(Write(explanation))
        self.wait(2)

class KalmanFilterVis(Scene):
    def construct(self):
        title = Text("4. Heteroscedastic Kalman Filter", font_size=36, color=YELLOW).to_edge(UP)
        self.add(title)

        # Main Axis
        ax = Axes(
            x_range=[0, 100, 10], y_range=[0, 1.5, 0.5],
            x_length=8, y_length=5,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)

        x_lbl = ax.get_x_axis_label("Bin / Distance")
        self.play(Create(ax), Write(x_lbl))

        # --- Step 1: Initial State (Previous Estimate) ---
        # Assume we have a prior estimate at 50 with some uncertainty
        mu_prev = 45
        sigma_prev = 8

        curve_prev = ax.plot(lambda x: norm.pdf(x, mu_prev, sigma_prev)*15, color=BLUE)
        area_prev = ax.get_area(curve_prev, color=BLUE, opacity=0.3)
        lbl_prev = Text("State (t-1)", color=BLUE, font_size=20).next_to(curve_prev, UL)

        self.play(Create(curve_prev), FadeIn(area_prev), Write(lbl_prev))
        self.wait(1)

        # --- Step 2: Incoming Measurement (Low Confidence / High Uncertainty) ---
        # A foggy reading, wide variance from MC Dropout
        mu_meas = 60
        sigma_meas = 10 # Very uncertain

        curve_meas = ax.plot(lambda x: norm.pdf(x, mu_meas, sigma_meas)*15, color=GREEN)
        area_meas = ax.get_area(curve_meas, color=GREEN, opacity=0.3)
        lbl_meas = Text("Measurement (t)\nHigh Uncertainty", color=GREEN, font_size=20).next_to(curve_meas, UR)

        self.play(Create(curve_meas), FadeIn(area_meas), Write(lbl_meas))

        # Show Math Concept
        math_text = MathTex(
            r"K_t = \frac{\sigma^2_{state}}{\sigma^2_{state} + \sigma^2_{meas}} \approx \text{Low}"
        ).to_edge(UR).scale(0.8).shift(DOWN*0.6)
        self.play(Write(math_text))

        # Update: Shift slightly towards measurement, but stay wide
        mu_new = 48 # Moved slightly from 45 towards 60
        sigma_new = 7 # Slightly tighter
        curve_upd = ax.plot(lambda x: norm.pdf(x, mu_new, sigma_new)*15, color=YELLOW)
        area_upd = ax.get_area(curve_upd, color=YELLOW, opacity=0.5)

        self.play(
            ReplacementTransform(curve_prev, curve_upd),
            ReplacementTransform(area_prev, area_upd),
            FadeOut(curve_meas), FadeOut(area_meas),
            FadeOut(lbl_prev), FadeOut(lbl_meas)
        )
        self.wait(1)

        # --- Step 3: Incoming Measurement (High Confidence / Low Uncertainty) ---
        # Clear signal, MC Dropout says variance is low
        mu_meas_2 = 40
        sigma_meas_2 = 2 # Very sharp

        curve_meas_2 = ax.plot(lambda x: norm.pdf(x, mu_meas_2, sigma_meas_2)*5, color=GREEN)
        area_meas_2 = ax.get_area(curve_meas_2, color=GREEN, opacity=0.3)
        lbl_meas_2 = Text("Measurement (t+1)\nHigh Confidence", color=GREEN, font_size=20).next_to(curve_meas_2, UP)

        self.play(Create(curve_meas_2), FadeIn(area_meas_2), Write(lbl_meas_2))

        # Update Math
        math_text_2 = MathTex(
            r"K_t \approx \text{High} \rightarrow \text{Trust Measurement}"
        ).to_edge(UR).scale(0.8).shift(DOWN*0.6)
        self.play(ReplacementTransform(math_text, math_text_2))

        # Final Update: Snaps to the high confidence measurement and gets sharp
        mu_final = 43.8
        sigma_final = 1.8 # Very tight estimate
        curve_final = ax.plot(lambda x: norm.pdf(x, mu_final, sigma_final)*5, color=YELLOW)
        area_final = ax.get_area(curve_final, color=YELLOW, opacity=0.8)
        lbl_final = Text("Final Estimate", color=YELLOW, font_size=24, weight=BOLD).next_to(curve_final, UP)

        self.play(
            ReplacementTransform(curve_upd, curve_final),
            ReplacementTransform(area_upd, area_final),
            FadeOut(curve_meas_2), FadeOut(area_meas_2),
            Transform(lbl_meas_2, lbl_final)
        )

        final_note = Text("Weighted by Inverse Variance", font_size=24, color=GRAY).to_corner(DR)
        self.play(Write(final_note))
        self.wait(3)
