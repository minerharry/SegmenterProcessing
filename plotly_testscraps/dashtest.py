

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from copy import copy
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
# from plotly_image_testing import fig
app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
print("wow")
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
print(fig)
fig = make_subplots(2,2,figure=fig)
print(fig)
fig.add_trace(px.bar(df,x="City",y="Fruit",color="Amount",barmode="group").data[0],1,2)
fig.add_trace(px.bar(df,x="City",y="Amount",color="City",barmode="group").data[0],2,1)
fig2 = make_subplots(1,3,figure=copy(fig))

fig.update_layout(transition={
                'duration': 500,
                'easing': 'cubic-in-out'
        })
fig2.update_layout(transition={
                'duration': 500,
                'easing': 'cubic-in-out'
        })

# fig2.data = fig2.data[::-1]
# 

app.layout = html.Div([
        dcc.Graph(
            id='example-graph',
            figure=fig),
        html.Button(
            children="B1",
            id='change-button',
            # value='b1'
            # text='b2'
        ),
        dcc.Dropdown(["f1","f2"],id="drop")
        ])
# app.

@app.callback(
    Output("example-graph","figure"),
    Input("change-button","n_clicks"),
    State("drop","value"),
    State("example-graph","figure")
)
def callback(wow,drop,f):
    print(go.Figure(f))
    if drop == "f2":
        # print(fig2)
        return fig2
    else:
        # print(fig)
        return fig
    # return 'b1'



if __name__ == '__main__':
    app.run_server(debug=True)