# Complex Activation Unit Disk Demo

This Panel app visualizes complex-valued activation functions on the unit disk.
It samples values where `|z| <= 1`, applies the selected activation from
`studies.activations.ACTIVATIONS`, and plots both the input disk and transformed
output with HoloViews on the Bokeh backend.

## Install

From this directory:

```powershell
pip install -r requirements.txt
```

The app requirements include the parent activation requirements plus:

- Bokeh
- HoloViews
- Panel
- Param

## Launch

From the repository root:

```powershell
python -m studies.activations.app.complex_activation_demo
```

The default server is:

```text
http://localhost:5006/
```

To choose another port:

```powershell
python -m studies.activations.app.complex_activation_demo --port 5007
```

Or use environment variables:

```powershell
$env:PANEL_PORT=5007
$env:PANEL_ADDRESS="localhost"
python -m studies.activations.app.complex_activation_demo
```

Add `--show` if you want Panel to ask Bokeh to open a browser window:

```powershell
python -m studies.activations.app.complex_activation_demo --show
```

## Interface

The sidebar contains:

- `Activation`: selects any function in the activation registry.
- `Resolution`: controls the unit disk sample density.
- `Point size`: changes the rendered point size.
- `Modrelu bias`: used when `modrelu` is selected.
- `Negative slope`: used when `split_leaky_relu` is selected.
- `Alpha`: used by `split_elu` and `split_selu`.
- `Beta`: used by `split_swish`.
- `Formulation`: shows the mathematical form for the selected activation.

The main view contains:

- Input plot: sampled complex values in the unit disk.
- Output plot: transformed values after the selected activation.
- Reference circle: the unit circle overlay for scale.

Both plots color points by complex phase and include Bokeh hover, pan, zoom,
save, and reset tools.

## Programmatic Use

```python
from studies.activations.app import ComplexActivationDemo, create_app

demo = ComplexActivationDemo(activation="cardioid")
panel_template = demo.panel()

app = create_app()
```

The module can also be served by Bokeh/Panel tooling because it exposes a
servable `app` object.

## Troubleshooting

If launch fails with `WinError 10048`, the selected port is already in use.
Start the app on a different port:

```powershell
python -m studies.activations.app.complex_activation_demo --port 5007
```
