# ClassCustomizedDA

## Introduction

This repository provides the official implementation of **Class-Customized Domain Adaptation (CCDA)**.  

The associated paper has been published in **IEEE Transactions on Image Processing (TIP)**:  
[Class-Customized Domain Adaptation: Unlock Each Customer-Specific Class with Single Annotation](https://ieeexplore.ieee.org/abstract/document/11142945)

## Environment Setup

Please ensure your development environment has the correct version of Python and all required dependencies installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Reproducing the experiments in the paper

- Download the corresponding open-source datasets and place them in the following directories:  
  - `./data/office31`  
  - `./data/officehome`  
  - `./data/domainnet`  

- Refer to `script.sh` for example commands to set datasets and parameters.

### 2. Running your own customization experiments

- Prepare your data:  
  - Save the paths of all labeled data into one `.txt` file.  
  - Save the paths of all unlabeled data into another `.txt` file.  
  - Place these files under `./data/list/`.

- Add a new dataset option in the code, for example:

```python
parser.add_argument(
    '--data_set',
    default='office_10_10',
    choices=['office_10_10', 'home_10_10', 'domainnet_10_10', 'office_5_15', 'office_15_5', 'office_3shot', 'fruit_5_5', 'dog_5_5'],
    help='data set'
)
```

- Set the corresponding number of shared classes and total categories:

```python
if config['data_set'] in ['office_10_10', 'home_10_10', 'domainnet_10_10', 'office_3shot']:
    pass
elif config['data_set'] == 'office_5_15':
    config['network']['params']['shared_class_num'] = 5
elif config['data_set'] == 'office_15_5':
    config['network']['params']['shared_class_num'] = 15
elif config['data_set'] in ['fruit_5_5', 'dog_5_5']:
    config['network']['params']['class_num'] = 10
    config['network']['params']['shared_class_num'] = 5
else:
    raise ValueError('dataset %s not found!' % (config['data_set']))
```

- Refer to `script.sh` to set your own `data_path` and parameters.  
  For the settings of the three `alpha_offs`, please check the **Implementation Details** section of the paper.

## Contributing

Contributions are welcome! You can submit issues or pull requests by following these steps:

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{chen2025class,
  title={Class-Customized Domain Adaptation: Unlock Each Customer-Specific Class With Single Annotation},
  author={Chen, Kaixin and Chang, Huiying and Xu, Mengqiu and Du, Ruoyi and Wu, Ming and Ma, Zhanyu and Zhang, Chuang},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```

## Contact

If you have any questions or suggestions, please feel free to reach out:  

📧 **Kaixin Chen** — chenkaixin@bupt.edu.cn  
