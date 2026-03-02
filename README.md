# `ImageGen2PowerSystems` 

*Have you ever wondered if you could use image generation deep learning models in the context of power system dynamic simulations?* 

I assume you haven't. We did.

This is the public repository (MIT License) for the model underlying our recent work regarding representational learning of time series. 
Our intuition was that powers system dynamic responses usually follow certain patterns, therefore, we could simply *tokenize*  them to ease downstream tasks such as modelling, forecasting and simulation using the wonderful VQVAE model. 

What started as simple experiments out of curiosity quickly turned into a full-blown project including several rabbit holes, manic night shifts and the first part will be presented at the **4th International Workshop on Open Source Modelling and Simulation of Energy Systems** in Karlsruhe, Germany in March 2026. 
The manuscript will be published here once available. 

## How to use?

In `scripts` we have simple experimental named `VQVAE_model_exp1.jl` that runs the experiments that are shown in the final manuscript. The models are shown in `models`, `data_generation` includes the training and validation data generation algorithms and the training loops can be found in `VQVAE_training` together with some helper functions in `VQVAE_utils.jl`. 
Together, this repo comprises the most essential core part of our current project.

## Contributors
- Florian Strebl (coding, writing; AIT, KTH) 
- Catalin Gavriluta (supervision; AIT)
- Qianwen Xu (supervision; KTH)

If any questions might arise or you wish to collaborate, contact <strebl@kth.se>


## License
The project is licensed under the general MIT License.


