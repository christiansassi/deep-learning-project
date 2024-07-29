# Deep Learning 2024 - Project Assignment
<img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>

## Introduction

Deep neural networks often suffer from severe performance degradation when tested on images that differ visually from those encountered during training. This degradation is caused by factors such as domain shift, noise, or changes in lighting.

Recent research has focused on domain adaptation techniques to build deep models that can adapt from an annotated source dataset to a target dataset. However, such methods usually require access to downstream training data, which can be challenging to collect.

An alternative approach is **Test-Time Adaptation (TTA)**, which aims to improve the robustness of a pre-trained neural network to a test dataset, potentially by enhancing the network's predictions on one test sample at a time. Two notable TTA methods for image classification are:

- **[Marginal Entropy Minimization with One test point (MEMO)](https://arxiv.org/pdf/2110.09506)**: This method uses pre-trained models directly without making any assumptions about their specific training procedures or architectures, requiring only a single test input for adaptation.
- **[Test-Time Prompt Tuning (TPT)](https://arxiv.org/pdf/2209.07511)**: This method leverages pre-existing models without any assumptions about their specific training methods or architectures, enabling adaptation using only a small set of labeled examples from the target domain.

## MEMO
For this project, MEMO was applied to a pretrained Convolutional Neural Network, **ViT-b/16**, using the **ImageNetV2** dataset. This network operates as follows: given a test point $x \in X$, it produces a conditional output distribution $p(y|x; w)$ over a set of classes $Y$, and predicts a label $\hat{y}$ as:

$$ \hat{y} = M(x | w) = \arg \max_{y \in Y} p(y | x; w) $$

<p align="center" text-align="center">
  <img width="75%" src="https://github.com/christiansassi/deep-learning-project/blob/main/assets/img1.jpg?raw=true">
  <br>
  <span><b>Fig. 1</b> MEMO overview</span>
</p>

Let $ A = \{a_1,...,a_M\} $ be a set of augmentations (resizing, cropping, color jittering etc...). Each augmentation $ a_i \in A $ can be applied to an input sample $x$, resulting in a transformed sample denoted as $a_i(x)$, as shown in figure. The objective here is to make the model's prediction invariant to those specific transformations.

MEMO starts by appling a set of $B$ augmentation functions sampled from $A$ to $x$. It then calculates the average, or marginal, output distribution $ \bar{p}(y | x; w) $ by averaging the conditional output distributions over these augmentations, represented as:

$$ \bar{p}(y | x; w) = \frac{1}{B} \sum_{i=1}^B p(y | a_i(x); w) $$

Since the true label $y$ is not available during testing, the objective of Test-Time Adaptation (TTA) is twofold: (i) to ensure that the model's predictions have the same label $y$ across various augmented versions of the test sample, (ii) to increase the confidence in the model's predictions, given that the augmented versions have the same label. To this end, the model is trained to minimize the entropy of the marginal output distribution across augmentations, defined as:

$$ L(w; x) = H(\bar{p}(\cdot | x;w)) = -\sum_{y \in Y} \bar{p}(y | x;w) \text{log} \bar{p}(y | x;w) $$

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/christiansassi/deep-learning-project
   cd deep_learning_project
   ```
2. Upload the notebook `deep_learning.ipynb` on [Google Colab](https://colab.research.google.com/). *NOTE: Make sure you use the T4 GPU*.

# Contacts

Matteo Beltrami - [matteo.beltrami-1@studenti.unitn.it](mailto:pietro.bologna@studenti.unitn.it)

Pietro Bologna - [luca.pedercini@studenti.unitn.it](mailto:luca.pedercini@studenti.unitn.it)

Christian Sassi - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<a href="https://www.unitn.it/"><img src="assets/unitn-logo.png" width="300px"></a>
