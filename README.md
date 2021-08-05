
# Learning-to-augment strategy
Learning-to-Augment Strategy Using Noisy and Denoised Data: An Algorithm to Improve Generalization of Deep CNN

# Abstract
Chest X-ray images are used in deep convolutional neural networks for the detection of COVID-19, the greatest human challenge of the 21st century. Robustness to noise and improvement of generalization are the major challenges in designing these networks. In this paper, we introduce a strategy for data augmentation using the determination of the type and value of noise density to improve the robustness and generalization of deep CNNs for COVID-19 detection. Firstly, we present a learning-to-augment approach that generates new noisy variants of the original image data with optimized noise density. We apply a Bayesian optimization technique to control and choose the optimal noise type and its parameters. Secondly, we propose a novel data augmentation strategy, based on denoised X-ray images, that uses the distance between denoised and original pixels to generate new data. We develop an autoencoder model to create new data using denoised images corrupted by the Gaussian and impulse noise. A database of chest X-ray images, containing COVID-19 positive, healthy, and non-COVID pneumonia cases, is used to fine-tune the pre-trained networks (AlexNet, ShuffleNet, ResNet18, and GoogleNet). The proposed method performs better results compared to the state-of-the-art learning to augment strategies in terms of sensitivity (0.808), specificity (0.915), and F-Measure (0.737).

# Keywords
Learning-to-augment, Data augmentation, Noise, X-ray images, Classification, COVID-19, Deep learning

# Cite
##### Please add the paper into reference if the repository is helpful to you.


Mohammad Momeny, Ali Asghar Neshat, Mohammad Arafat Hussain, Solmaz Kia, Mahmoud Marhamati, Ahmad Jahanbakhshi, Ghassan Hamarneh,
**Learning-to-Augment Strategy using Noisy and Denoised Data: Improving Generalizability of Deep CNN for the Detection of COVID-19 in X-ray Images,**
*Computers in Biology and Medicine*,
Volume 136, 2021, 104704,
ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2021.104704.
(https://www.sciencedirect.com/science/article/pii/S0010482521004984)

# The source code
The following Matlab code is the original code corresponding to the paper, Learning-to-Augment Strategy Using Noisy and Denoised Data: An Algorithm to Improve Generalization of Deep CNN for the Detection of COVID-19 in X-ray Images.

# Email
Feel free to ask any questions. Dr. Mohammad Momeny, mohamad.momeny@gmail.com
