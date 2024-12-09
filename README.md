<div align="center">

<h1>&#128293;  HumanSplat: Generalizable Single-Image Human
Gaussian Splatting with Structure Priors</h1>

 [![Arxiv](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.12459)  [![Project Page](https://img.shields.io/badge/HumanSplat-project-red?logo=googlechrome&logoColor=blue)](https://humansplat.github.io/) [![Code](https://img.shields.io/badge/HumanSplat-Code-red?logo=googlechrome&logoColor=blue)](https://github.com/humansplat/humansplat) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

<video> 
   <source src="./data/assets/humansplat.mp4" type="video/mp4">
</video>


<strong> HumanSplat predicts 3D Gaussian Splatting properties from a single input image in a generalizable manner.</strong>
<!-- --- -->
</div>



### 📊  Dataset
Download human datasets ([Thuman2.0](https://github.com/ytrock/THuman2.0-Dataset), [2K2K](https://github.com/SangHunHan92/2K2K) and [Twindom](https://web.twindom.com/)) and organize them as follows:
```bash
${ROOT} 
├──📂data/
    ├──📂Thuman2/
       ├── 📂0000/
       ├── 📂...
       ├── 📂0525/
    ├──📂2K2K/
       ├── 📂00003/
       ├── 📂...
       ├── 📂04739/
    ├──📂Twindom/

```

### 🔧  Installation

Additionally dependencies include: 
#### Register an username & password for [Website](https://icon.is.tue.mpg.de/) is required.

```bash

bash settings/fetch_hps.sh
```


Additionally dependencies include: 

```bash
bash settings/setup.sh
```



###  Run `./demo.py` 
```bash
python3 demo.py --config xxx.yaml 
```



## 📚 Citation
 If you find our work useful for your research, please consider citing:
```bibtex
@article{pan2024humansplat,
  title={HumanSplat: Generalizable Single-Image Human Gaussian Splatting with Structure Priors},
  author={Pan, Panwang and Su, Zhuo and Lin, Chenguo and Fan, Zhen and Zhang, Yongjie and Li, Zeming and Shen, Tingting and Mu, Yadong and Liu, Yebin},
  journal={arXiv preprint arXiv:2406.12459},
  year={2024}
}
```






