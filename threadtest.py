import matplotlib.pyplot as plt
from IPython import embed

# Set the backend to Qt
plt.switch_backend('Qt5Agg')

# Enable interactive mode in Matplotlib
plt.ion()

# Create your plot
plt.plot([1,2,3,4],[1, 2, 3, 4])
plt.xlabel('X')
plt.ylabel('Y')

# Show the initial plot
plt.show()

# Embed an interactive terminal
embed()