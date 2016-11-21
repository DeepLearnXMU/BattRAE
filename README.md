# BattRAE
source code for "BattRAE: Bidimensional Attention-Based Recursive Autoencoders for Learning Bilingual Phrase Embeddings"

If you use this code, please cite our paper:
```
@InProceedings{Zhang:AAAI:2017:BattRAE,
  author    = {Biao Zhang, Deyi Xiong and Jinsong Su},
  title     = {BattRAE: Bidimensional Attention-Based Recursive Autoencoders for Learning Bilingual Phrase Embeddings},
  booktitle = {Proc. of AAAI},
  year      = {2017},
}
```

## Basic Requirement

1. Eigen, http://eigen.tuxfamily.org/index.php
2. lbfgs, http://www.chokkan.org/software/liblbfgs/
3. g++
4. Linux x86_64

## How to Run?

### To compile the program, use the command
```
make
```

### To run the program, use the command
1. training
```
    ./battrae-model Config.ini -train
```
2. testing
```
    ./battrae-model Config.ini -test
```

A demo example is given in directory: demo/, there are one subdirectory:
    data/       the training data, test data and dev corpus, with pretrained worde embeddings
The format of training/dev data:
    correct source phrase ||| correct target phrase ||| negative source phrase ||| negative target phrase
See detail in the demo example.

Example outputs are given in the demo, and see the "Config.ini" for more detailed training and test settings.

For any comments or questions, please email <a href="mailto:zb@stu.xmu.edu.cn">Biao Zhang</a>.
