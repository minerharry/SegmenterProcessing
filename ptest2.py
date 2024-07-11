import plotly.graph_objects as go
from plotly.subplots import make_subplots
import IPython

# Create subplots
fig = make_subplots(rows=2, cols=3)

IPython.embed()

# Define data for each subplot
data = [
    [1, 2, 3],
    [3, 2, 1]
]

# Add initial traces to subplots
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Trace 1'), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Trace 2'), row=1, col=2)

# Define animation frames
frames = []
for i in range(len(data[0])):
    frame = go.Frame(
        data=[
            go.Scatter(y=[0, data[0][i]], x=[0, 1], mode='lines', name='Trace 1'),
            go.Scatter(y=[0, data[1][i]], x=[0, 1], mode='lines', name='Trace 2')
        ]
    )
    frames.append(frame)

# Update frames in the figure
fig.frames = frames

# Configure animation button
animation_button = dict(
    label='Play',
    method='animate',
    args=[None, {'frame': {'duration': 1000, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 500, 'easing': 'linear'}}]
)

# Add animation button to the figure
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[animation_button]
        )
    ]
)

print(fig)
IPython.embed()

# Show the figure
fig.show()