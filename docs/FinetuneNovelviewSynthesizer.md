

### Acknowledgement

We would like to thank [sv3d-diffusers](https://github.com/chenguolin/sv3d-diffusers)'s [author](https://github.com/chenguolin) for open-sourcing their implementation.


## 🚀 Usage
```bash
git clone https://github.com/chenguolin/sv3d-diffusers.git
# Please install PyTorch first according to your CUDA version
pip3 install -r requirements.txt

# If you can't access to HuggingFace🤗, try:
# export HF_ENDPOINT=https://hf-mirror.com
python3 infer.py --output_dir out/ --image_path assets/images/sculpture.png --elevation 10 --half_precision --seed -1
```
The synthesized video will save at `out/` as a `.gif` file.

Converted SV3D-p checkpoints have been uploaded to HuggingFace🤗 [chenguolin/sv3d-diffusers](https://huggingface.co/chenguolin/sv3d-diffusers).
