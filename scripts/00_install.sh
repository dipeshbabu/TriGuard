python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip

# Pick the right torch build for your machine; CUDA example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install numpy matplotlib
pip install auto_LiRPA
