# Global constants
FLOAT_TYPE_NAME = "float32"  # Recommended: "float32" for faster computation
PITCH_DEFAULT = 0.5

# Taichi backend selection
# Options: "auto", "cuda", "vulkan", "metal", "opengl", "cpu"
# "auto" will try backends in order: cuda -> vulkan -> metal -> opengl -> cpu
TAICHI_BACKEND = "auto"
