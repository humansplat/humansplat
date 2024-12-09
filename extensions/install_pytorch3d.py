import os
import sys
import torch

if __name__ == "__main__":
    pyt_version_str=torch.__version__.split("+")[0].replace(".", "")

    version_str="".join([
        f"py3{sys.version_info.minor}",
        torch.version.cuda.replace(".",""),
        f"_pyt{pyt_version_str}"
    ])

    os.system('pip3 install fvcore iopath')
    os.system('pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/%s/download.html' % version_str)
