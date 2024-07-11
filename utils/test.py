from animated_slider import PlaybackSliderAIO
from dash import Dash, html, callback, Output, Input
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ]
)
app.layout = html.Div([
    PlaybackSliderAIO(
        aio_id='bruh',
        slider_props={'min': 0, 'max': 10, 'step': 1, 'value': 0},
        button_props={'className': 'float-left'}
    ),
    html.Div(id='text')
])


@callback(
    Output('text', 'children'),
    Input(PlaybackSliderAIO.ids.slider('bruh'), 'value')
)
def update_children(val):
    return val


if __name__ == "__main__":
    app.run_server(debug=True,port=8049)
