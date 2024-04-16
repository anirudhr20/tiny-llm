# TinyLLM

## Description
TinyLLM is a lightweight implementation of a decoder model, along with a tokenizer, using PyTorch. This project was inspired by Andrej Karpathy's lecture on language modeling.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description
TinyLLM aims to provide a simple and accessible implementation of a decoder model for natural language processing tasks. It utilizes PyTorch, a popular deep learning framework, to build and train the model.

## Installation
To install and set up TinyLLM, follow these steps:
1. Clone the repository: 

`$ git clone https://github.com/anirudhr20/tiny-llm.git`

2. Install the required dependencies: 

`$ pip install -r requirements.txt`

## Usage
To use TinyLLM in your own projects, follow these guidelines:
1. Import the necessary modules: 
- `$ from llm.model import TinyLLM`
- `$ from llm.model import TinyTokenizer`
2. Training the model:
`cd Scripts/ python train.py`

3. Model Inference:
`cd Scripts/ python inference.py`

## Contributing
Contributions to TinyLLM are welcome! If you would like to contribute, please follow these steps:
1. Fork the repository: [https://github.com/anirudhr20/tiny-llm.git](https://github.com/anirudhr20/tiny-llm.git)
2. Create a new branch for your feature: `$ git checkout -b feature-name`
3. Make your changes and commit them: `$ git commit -m "Add feature-name"`
4. Push your changes to your forked repository: `$ git push origin feature-name`
5. Open a pull request on the main repository.

## License
TinyLLM is distributed under the MIT License. See the [LICENSE](https://github.com/anirudhr20/tiny-llm/blob/main/LICENSE) file for more information.
