"""Interactive complex activation demo for Panel/Bokeh."""

from __future__ import annotations

import argparse
import os

import holoviews as hv
import numpy as np
import panel as pn
import param

from studies.activations import ACTIVATIONS


hv.extension("bokeh")
pn.extension()


FORMULAS = {
    "amplitude_sigmoid": r"$f(z)=\sigma(|z|)\frac{z}{|z|+\epsilon}$",
    "amplitude_tanh": r"$f(z)=\tanh(|z|)\frac{z}{|z|+\epsilon}$",
    "cardioid": r"$f(z)=\frac{1}{2}(1+\cos(\arg z))z$",
    "complex_sigmoid": r"$f(z)=\frac{1}{1+e^{-z}}$",
    "complex_tanh": r"$f(z)=\tanh(z)$",
    "crelu": r"$f(z)=\operatorname{ReLU}(\operatorname{Re}z)+i\operatorname{ReLU}(\operatorname{Im}z)$",
    "identity": r"$f(z)=z$",
    "modrelu": r"$f(z)=\operatorname{ReLU}(|z|+b)\frac{z}{|z|+\epsilon}$",
    "split_elu": r"$f(z)=\operatorname{ELU}(\operatorname{Re}z)+i\operatorname{ELU}(\operatorname{Im}z)$",
    "split_gelu": r"$f(z)=\operatorname{GELU}(\operatorname{Re}z)+i\operatorname{GELU}(\operatorname{Im}z)$",
    "split_leaky_relu": (
        r"$f(z)=\operatorname{LeakyReLU}(\operatorname{Re}z)"
        r"+i\operatorname{LeakyReLU}(\operatorname{Im}z)$"
    ),
    "split_relu": r"$f(z)=\operatorname{ReLU}(\operatorname{Re}z)+i\operatorname{ReLU}(\operatorname{Im}z)$",
    "split_selu": r"$f(z)=\operatorname{SELU}(\operatorname{Re}z)+i\operatorname{SELU}(\operatorname{Im}z)$",
    "split_sigmoid": r"$f(z)=\sigma(\operatorname{Re}z)+i\sigma(\operatorname{Im}z)$",
    "split_softplus": (
        r"$f(z)=\operatorname{softplus}(\operatorname{Re}z)"
        r"+i\operatorname{softplus}(\operatorname{Im}z)$"
    ),
    "split_swish": r"$f(z)=\operatorname{swish}_{\beta}(\operatorname{Re}z)+i\operatorname{swish}_{\beta}(\operatorname{Im}z)$",
    "split_tanh": r"$f(z)=\tanh(\operatorname{Re}z)+i\tanh(\operatorname{Im}z)$",
    "zrelu": r"$f(z)=z\ \mathrm{if}\ \operatorname{Re}z\geq0,\operatorname{Im}z\geq0;\ 0\ \mathrm{otherwise}$",
}


def _unit_disk(resolution: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution)
    real, imag = np.meshgrid(axis, axis)
    z = real + 1j * imag
    return z[np.abs(z) <= 1.0].ravel()


def _circle(samples: int = 360) -> hv.Curve:
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    return hv.Curve((np.cos(theta), np.sin(theta)), kdims=["real"], vdims=["imag"]).opts(
        color="#1f2937",
        line_width=2,
        alpha=0.8,
    )


class ComplexActivationDemo(param.Parameterized):
    """Param-powered explorer for complex-valued activation functions."""

    activation = param.ObjectSelector(default="cardioid", objects=sorted(ACTIVATIONS))
    resolution = param.Integer(default=81, bounds=(25, 181), step=2)
    point_size = param.Integer(default=5, bounds=(2, 10))
    modrelu_bias = param.Number(default=-0.25, bounds=(-1.0, 1.0), step=0.05)
    negative_slope = param.Number(default=0.01, bounds=(0.0, 0.5), step=0.01)
    alpha = param.Number(default=1.0, bounds=(0.05, 3.0), step=0.05)
    beta = param.Number(default=1.0, bounds=(0.05, 5.0), step=0.05)

    width = param.Integer(default=430, precedence=-1)
    height = param.Integer(default=430, precedence=-1)

    def _activation_kwargs(self) -> dict[str, float]:
        if self.activation == "modrelu":
            return {"bias": self.modrelu_bias}
        if self.activation == "split_leaky_relu":
            return {"negative_slope": self.negative_slope}
        if self.activation == "split_elu":
            return {"alpha": self.alpha}
        if self.activation == "split_selu":
            return {"alpha": self.alpha}
        if self.activation == "split_swish":
            return {"beta": self.beta}
        return {}

    def _activated(self, z: np.ndarray) -> np.ndarray:
        activation = ACTIVATIONS[self.activation]
        return activation(z, **self._activation_kwargs())

    def _points(self, z: np.ndarray, label: str) -> hv.Points:
        data = {
            "real": np.real(z),
            "imag": np.imag(z),
            "magnitude": np.abs(z),
            "phase": np.angle(z),
        }
        return hv.Points(data, kdims=["real", "imag"], vdims=["magnitude", "phase"], label=label).opts(
            cmap="Viridis",
            color="phase",
            colorbar=True,
            framewise=True,
            marker="circle",
            size=self.point_size,
            tools=["hover"],
            width=self.width,
            height=self.height,
            aspect="equal",
            xlabel="Real",
            ylabel="Imaginary",
            show_grid=True,
            toolbar="above",
        )

    @param.depends(
        "activation",
        "resolution",
        "point_size",
        "modrelu_bias",
        "negative_slope",
        "alpha",
        "beta",
    )
    def plot(self) -> hv.Layout:
        z = _unit_disk(self.resolution)
        activated = self._activated(z)
        circle = _circle()

        input_plot = (self._points(z, "Input unit disk") * circle).opts(
            title="Input z, |z| <= 1",
            xlim=(-1.1, 1.1),
            ylim=(-1.1, 1.1),
        )
        output_limit = max(1.1, float(np.nanmax(np.abs(activated))) * 1.1)
        output_plot = (self._points(activated, f"{self.activation}(z)") * circle).opts(
            title=f"Activation output: {self.activation}",
            xlim=(-output_limit, output_limit),
            ylim=(-output_limit, output_limit),
        )
        return (input_plot + output_plot).cols(2)

    def panel(self) -> pn.template.FastListTemplate:
        controls = pn.Param(
            self,
            parameters=[
                "activation",
                "resolution",
                "point_size",
                "modrelu_bias",
                "negative_slope",
                "alpha",
                "beta",
            ],
            widgets={
                "activation": pn.widgets.Select,
                "resolution": pn.widgets.IntSlider,
                "point_size": pn.widgets.IntSlider,
                "modrelu_bias": pn.widgets.FloatSlider,
                "negative_slope": pn.widgets.FloatSlider,
                "alpha": pn.widgets.FloatSlider,
                "beta": pn.widgets.FloatSlider,
            },
            show_name=False,
        )

        formula = pn.Column(
            pn.pane.Markdown("### Formulation", margin=(0, 0, 4, 0)),
            pn.pane.LaTeX(self.formula, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            margin=(16, 0, 0, 0),
        )

        template = pn.template.FastListTemplate(
            title="Complex Activation Unit Disk Demo",
            sidebar=[controls, formula],
            main=[pn.panel(self.plot, sizing_mode="stretch_width")],
            accent_base_color="#2563eb",
            header_background="#111827",
        )
        return template

    @param.depends("activation")
    def formula(self) -> str:
        return FORMULAS[self.activation]


def create_app() -> pn.template.FastListTemplate:
    return ComplexActivationDemo().panel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the complex activation unit disk demo.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PANEL_PORT", "5006")),
        help="Bokeh server port. Defaults to PANEL_PORT or 5006.",
    )
    parser.add_argument(
        "--address",
        default=os.environ.get("PANEL_ADDRESS", "localhost"),
        help="Bokeh server address. Defaults to PANEL_ADDRESS or localhost.",
    )
    parser.add_argument("--show", action="store_true", help="Open the app in a browser after launch.")
    args = parser.parse_args()
    pn.serve({"/": create_app}, port=args.port, address=args.address, show=args.show, autoreload=False)


app = create_app()


if __name__.startswith("bokeh"):
    app.servable()


if __name__ == "__main__":
    main()
