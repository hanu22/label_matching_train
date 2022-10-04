import base64
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

logo_image = Path(
    "images",
    "ds_logo.png",
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

ds_logo_encoded = base64.b64encode(
    open(
        logo_image,
        "rb",
    ).read()
)

img_ds = html.Img(
    src=f"data:image/png;base64,{ds_logo_encoded.decode()}",
    style={
        "float": "left",
        "width": "50%",
    },
)

title = "PROJECT TITLE"

app.layout = dbc.Container(
    [
        html.Div(
            [
                # row1: logos
                dbc.Row(
                    [
                        dbc.Col(img_ds),
                        dbc.Col(
                            title,
                        ),
                    ],
                    align="center",
                    className="mb-2",
                ),
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port="8080",
    )

# python app.py
