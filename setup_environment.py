import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "gymnasium",           # OpenAI Gym
    "torch",         # PyTorch, you can replace it with 'tensorflow' if you prefer
    "numpy",         # NumPy
    "matplotlib",     # Matplotlib
    'gym[box2d]',    # Box2D
    'pygame',        # Pygame
]

# Installing packages
for package in required_packages:
    install(package)

print("All required packages are installed.")
