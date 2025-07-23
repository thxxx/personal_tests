apt-get update
git config --global user.email zxcv05999@naver.com
git config --global user.name thxxx
python -m pip install --upgrade pip
pip install tqdm matplotlib einops diffusers accelerate transformers datasets opencv-python torchdiffeq clean-fid jaxtyping tensorboard
pip install --upgrade pillow

pip uninstall torchaudio
pip install "numpy<2"

# wget https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true