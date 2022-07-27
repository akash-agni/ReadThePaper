# Understanding Deep Convolutional GAN

In 2014 Ian Goodfellow et al. proposed a new approch for estimation of generative models through adverarial process. It involved training two seperate models at the same time, a Generator model which attempts to model the data distribution and ad Discriminator which attempts to classify the input as either training data or fake data by generator.

The paper sets a very important milestone in modern machine learning landscape, opening new avenues for unsupervised learning. Deep Convolutional GAN (Radford et al. 2015) continued building on this idea by applying the principles on a convolutional network to produce 2D images succesfully.

## What is so cool about GAN.
To understand the importance of GAN or DCGAN lets look at what makes them so popular.

<ol>
    <li>As a large percentage of real world data is unlabbeled the unsupervised learning nature of GAN's make them ideal for such use cases.</li>
    <li>Generator and Discriminator act as very good feature extractor for uses cases with limited labelled data or generate additional data ti impove secondry model training, as it can generate fake samples instead of using augmentations.</li>
    <li>GAN's provide and alternative to maximum likelihood techniques. Their adversarial learning process and non heuristic cost function makes them very attractive to reinforcement learning.</li>
    <li>The research around GAN has been very attractive and the results have been source of widespread debate on the impact ML/DL. For example DeepFake, one of the applications of GAN which can overlay people's face on a target person, has been very controverisal in nature as it has the potential to be used for nefarious purposes.</li>
    <li>The last but the most important point being, its just so very cool to work with, and all the new research in the feild has been mesmerizing.</li>
</ol>

## Architecture

<img src="../images/DCGAN_Arch.png">Architecture of DCGAN</img>

As we discussed earlier, we will be working through DCGAN which attempts to implement the core ideas of GAN for a convolutional network which can generate realistic looking images.

DCGAN is made up of two seperate models, a Generator (G) which attempts to model which takes a random noise vector as input and attempts learn data disributionan to generate fake samples and a Discriminator (D) which takes training data (real samples) and generated data (fake samples) and tries to classify them, this struggle between the two models is what we call adversarial training process where one's loss is other's benifit.

## Generator

Generator is the one we are most intreasted in as it is the one generating fake images to try and fool the discriminator.

Now lets look at the generator achitecture in more detail.
Generator takes a noise vector as input and applies it over a linear layer, the size of noise dimension 

