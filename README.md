# ClassCustomizedDA

## Introduction

ClassCustomizedDA is the codebase of the CCDA method, which addresses model customization tasks involving customer-specific classes with just one additional annotation per customer-specific class.

The associated paper, **Class-Customized Domain Adaptation: Unlock Each Customer-Specific Class with Single Annotation**, is currently under review.

## Environment Setup

Ensure your development environment meets the necessary requirements, including the correct version of Python and the required dependencies.

    pip install -r requirements.txt


## Quick Start

1. **The customization experiments described in the paper:**

   -  Please download the corresponding open-source datasets and place them in the appropriate directories. (./data/office31, ./data/officehome, and ./data/domainnet)
   - Refer to the comments and commands in script.sh to set the data and parameters you wish to use. 

2. **Your own customization experiments:**

   - Please refer to the files under './data/list/', save the addresses of all labeled data to one '.txt' file, and save the addresses of all unlabeled data to another '.txt' file.

   - Add a new data_set option:

      ```python
      parser.add_argument('--data_set', default='office_10_10', choices=['office_10_10', 'home_10_10', 'domainnet_10_10', 'office_5_15', 'office_15_5', 'office_3shot', 'fruit_5_5', 'dog_5_5'], help='data set')

   - Set the corresponding number of shared classes and the total number of categories.

      ```python
      if config['data_set'] == 'office_10_10' or config['data_set'] == 'home_10_10' or   config['data_set'] == 'domainnet_10_10' or config['data_set'] == 'office_3shot':
          pass
      elif config['data_set'] == 'office_5_15':
          config['network']['params']['shared_class_num'] = 5   
      elif config['data_set'] == 'office_15_5':
          config['network']['params']['shared_class_num'] = 15            
      elif config['data_set'] == 'fruit_5_5' or config['data_set'] == 'dog_5_5':
          config['network']['params']['class_num'] = 10
          config['network']['params']['shared_class_num'] = 5
      else:
          raise ValueError('dataset %s not found!' % (config['data_set']))     

   - Refer to the comments and commands in script.sh to set your own data_path and parameters. For the settings of the three aplha_offs, please refer to the _Implementation Details_ section in the paper.


## Contributing and Contact
Contributions are welcome! You can contribute by submitting issues or pull requests.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

If you have any questions or suggestions, please open an issue or contact us directly.