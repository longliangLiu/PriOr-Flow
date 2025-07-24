# 🥳 PriOr-Flow（ICCV 2025 Highlight）🥳

This repository contains the source code for our paper:

PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View. <a href="https://arxiv.org/abs/2506.23897"><img src="https://img.shields.io/badge/arXiv-2506.23897-b31b1b?logo=arxiv" alt='arxiv'></a>

Longliang Liu, Miaojie Feng, Junda Cheng, Jijun Xiang, Xuan Zhu, Xin Yang

![Overview](./PriOr-RAFT/media/PriOr-RAFT.png)

## ⚙️ Installation
* NVIDIA RTX 3090
* python 3.8

### ⏳ Create a virtual environment and activate it.

```Shell
conda create -n priorflow python=3.8
conda activate priorflow
```

### 🎬 Dependencies
```Shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install matplotlib 
pip install tqdm
pip install wandb
```

## 💾 Required Data

* [MPFDataset](https://github.com/HenryLee0314/ECCV2022-MPF-net)
* [FlowScape](https://github.com/MasterHow/PanoFlow)
* [OmniPhotos](https://github.com/cr333/OmniPhotos)
* [ODVista](https://github.com/Omnidirectional-video-group/ODVista)

## 🧬 Model weights

| Model                       |          Link          |
|:----------------------------|:----------------------:|
| PriOr-RAFT(MPFDataset-EFT)  | [Download](https://drive.google.com/file/d/1QJuBMlPR1IZsqf__SO9E5YIxGKPKanC6/view?usp=drive_link) |
| PriOr-RAFT(MPFDataset-City) | [Download](https://drive.google.com/file/d/10Npvy3Oea92-pN9jNNKr5Wu3yO6JwQJZ/view?usp=drive_link) |
| PriOr-RAFT(FlowScape)       | [Download](https://drive.google.com/file/d/1xP9tONXOiQelYtNJbgxml4PafenCWVeq/view?usp=drive_link) |

**Here, we use PriOr-RAFT as an example to illustrate the evaluation and training process.**
``` Shell
cd PriOr-RAFT
```

## 🛴 Demos
You can quickly test the model forward pass using the provided `demo.py` script:
``` Shell
python demo.py
```
You can run inference using the pretrained PriOr-RAFT model on your own image pairs. A demo script is provided to visualize the predicted panoramic optical flow:
``` Shell
python demo_image.py
```

## 🧪 Evaluation
You can evaluate a trained model using `evaluate.py`:
``` Shell
python evaluate.py --model ./checkpoints/EFT/EFT-final.pth --dataset MPFDataset --scene EFT
```
Please see `./scripts` for more evaluation scripts.
You can also evaluate the model indifferent regions:
``` Shell
python evaluate.py --model ./checkpoints/EFT/EFT-final.pth --dataset MPFDataset --scene EFT --regions
```

## 🍲 Training
To train PriOr-RAFT on MPFDataset or FlowScape, you need to first download the RAFT pre-trained weights on FlyingThings and place them in the `./pretrained` directory, and then run:

``` Shell
./scripts/train_EFT.sh
./scripts/train_City.sh
./scripts/train_FlowScape.sh
```

## 📚 Citation
If you find our works useful in your research, please consider citing our papers:

```bibtex
@misc{liu2025priorflow,
      title={PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View}, 
      author={Longliang Liu and Miaojie Feng and Junda Cheng and Jijun Xiang and Xuan Zhu and Xin Yang},
      year={2025},
      eprint={2506.23897},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23897}, 
}
```

## 📧 Contact
Please feel free to contact me (Longliang) at [longliangl@hust.edu.cn](longliangl@hust.edu.cn).

# Acknowledgements
This project is based on [RAFT](https://github.com/princeton-vl/RAFT), [GMA](https://github.com/zacjiang/GMA), [SKFlow](https://github.com/littlespray/SKFlow) and [MonSter](https://github.com/Junda24/MonSter). We thank the original authors for their excellent works. 