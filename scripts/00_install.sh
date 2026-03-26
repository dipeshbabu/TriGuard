python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install ./auto_LiRPA
