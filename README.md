# Read The Paper
Reading papers is one of the most important habits to fall into if one wishes to be a successful Machine Learning Engineer, an advice I have ignored for a long time.
But its never too late, I have picked the habit of auditing as many papers as I can, but once in a while someone puts out work which redefines the landscape and through this repo my intention is to implement such milestone papers through `python`.

* 1 -[Attention is All You Need](https://github.com/akash-agni/ReadThePaper/blob/main/Attention%20Is%20All%20You%20Need.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akash-agni/LearningLanguage/blob/main/Attention%20Is%20All%20You%20Need.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-1706.03762-f9f107.svg)](https://arxiv.org/abs/1706.03762)

    <i>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin</i>

    This paper introduced the concept of <b>Transformers</b> which has since been the leading architecture for most Sequence-to-Sequence model like <b>BERT</b> and <b>GPT-2</b>.

    Before this RNN based model were dominent, but the big problem with them was they were unable to solve long term dependecies while training, this limited their usage over long sequences.


* 2 -[Deep Convolutional Generative Adversial Network](https://github.com/akash-agni/ReadThePaper/blob/main/DCGAN/dcgan.py) [![arXiv](https://img.shields.io/badge/arXiv-1511.06434-b31b1b.svg)](https://arxiv.org/abs/1511.06434)

    <i>Alec Radford, Luke Metz and Soumith Chintala</i>

    In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications.
    
    Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. 
    
    Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

## References
- https://github.com/bentrevett/pytorch-seq2seq
- https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
- https://github.com/eriklindernoren/PyTorch-GAN/tree/36d3c77e5ff20ebe0aeefd322326a134a279b93e
