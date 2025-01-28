# Q-Learning Maze Solver 🧠⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.1.3-660000.svg)](https://www.pygame.org/)

An intelligent agent that learns to navigate complex mazes using Q-learning algorithm, featuring real-time visualization and adaptive environment generation.

![Image](https://github.com/user-attachments/assets/33d950fa-d4fc-45e9-9465-5cd56cba0931)

## Features 🚀

- **Reinforcement Learning Core**
  - Q-learning implementation with ε-greedy exploration
  - Dynamic reward system (+100 goal, -50 wall collision)
  - Experience replay through persistent wall memory

- **Dynamic Environment**
  - Procedural maze generation (45% walls)
  - BFS-validated paths guarantee solvability
  - Persistent wall memory across episodes

- **Visualization Engine**
  - Real-time Pygame rendering
  - Blue path tracing for agent's trajectory
  - Clear visual hierarchy (GRAY walls, RED agent, GREEN target)

- **Performance Optimization**
  - Configurable grid sizes (30x30 default)
  - Adjustable learning parameters (ε decay: 0.995)
  - High-speed rendering (200 FPS cap)

## Installation 📦

1. **Requirements**:
   ```bash
   Python 3.8+
   Pygame 2.1.3
   NumPy
Install dependencies:

bash
```
pip install pygame numpy
```
Usage ▶️
```
python3 agent.py
```


License 📄
This project is licensed under the MIT License - see the LICENSE file for details.
