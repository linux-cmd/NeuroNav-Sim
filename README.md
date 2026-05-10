# NeuroNav-Sim

**NeuroNav-Sim** is a beginner-friendly autonomous navigation simulator built with Python.

It demonstrates robotics, computer vision, mapping, path planning, and robot control without requiring:

- a physical robot
- ROS
- Linux
- a GPU
- downloaded AI models
- expensive hardware

The project runs directly from VS Code using:

```bash
python app.py
```

Controls:

- Mouse click inside the world to set a goal
- Space to force a replan
- G to place a random goal
- N/P to switch difficulty levels
- 1-4 to jump directly to a level
- R to reset the current level

The robot uses sensor-fusion mapping, risk-aware A* planning, frontier exploration, and harder built-in maze levels.
