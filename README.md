### Towards Machine Unlearning Benchmarks: Forgetting the Personal Identities in Facial Recognition Systems
* This repository provides **practical benchmark datasets** and **PyTorch implementations for Machine Unlearning**, enabling the construction of privacy-crucial AI systems by forgetting specific data instances without changing the original model utility.

### Authors
[Dasol Choi](https://github.com/Dasol-Choi), [Dongbin Na](https://github.com/ndb796)

#### Abstract
> Machine unlearning is a crucial tool for enabling a classification model to forget specific data that are used in the training time. Recently, various studies have presented machine unlearning algorithms and evaluated their methods on several datasets. However, most of the current machine unlearning algorithms have been evaluated solely on traditional computer vision datasets such as CIFAR-10, MNIST, and SVHN. Furthermore, previous studies generally evaluate the unlearning methods in the class-unlearning setup. Most previous work first trains the classification models and then evaluates the machine unlearning performance of machine unlearning algorithms by forgetting selected image classes (categories) in the experiments. Unfortunately, **these class-unlearning settings might not generalize to real-world scenarios.** In this work, we propose a machine unlearning setting that aims to unlearn specific instance that contains personal privacy (identity) **while maintaining the original task of a given model.** Specifically, we propose two machine unlearning benchmark datasets, **MUFAC** and **MUCAC**, that are greatly useful to evaluate the performance and robustness of a machine unlearning algorithm. In our benchmark datasets, the original model performs facial feature recognition tasks: face age estimation (multi-class classification) and facial attribute classification (binary class classification), where a class does not depend on any single target subject (personal identity), which can be a realistic setting. Moreover, we also report **the performance of the state-of-the-art machine unlearning methods on our proposed benchmark datasets.**

#### <b>Task-Agnostic Machine Unlearning</b>

* The conceptual illustration of our proposed task-agnostic unlearning setup:

<img src="./resources/image_2.jpg" width=720px/>

* Comparison with the traditional class-unlearning:

<img src="./resources/image_1.jpg" width=720px/>

#### Datasets

* The illustration of our MUFAC benchmark:

<img src="./resources/image_3.jpg" width=360px/>

* [**MUFAC** (Machine Unlearning for Facial Age Classifier)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg)
  * A multi-class age classification dataset based on AI HUB, featuring over 13,000 Asian facial images with annotations for age groups and personal identities, ideal for machine unlearning research.
 <p align="center"><img src="./resources/MUFAC.png" style="width: 90%;"/></p>
 
* [**MUCAC** (Machine Unlearning for Celebrity Attribute Classifier)](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch)
  * A multi-label facial attribute classification dataset based on CelebA, expanded to 30,000 images and enriched with personal identity annotations to support unlearning algorithms.
 <p align="center"><img src="./resources/MUCAC.png" style="width: 90%; ;"/></p>

#### Evaluation Metrics

Our machine unlearning Benchmark is evaluated on two key aspects: **model utility** and **forgetting performance**. Here's how we measure them:

* **Model Utility**
  - **Accuracy**: The primary metric for model utility is the accuracy of the classification task, defined as the probability of the model's predictions matching the true labels in the test dataset.

* **Forgetting Score**
  - **Membership Inference Attack (MIA)**: To assess how well the model forgets, we use MIA, where the goal is to infer if specific data was used during training. A binary classifier is trained to distinguish between (1) data to be forgotten and (2) unseen data, with the ideal accuracy being 0.5, indicating perfect unlearning.

* **Normalized Machine Unlearning Score (NoMUS)**
  - **Combined Metric**: NoMUS is introduced to evaluate unlearning performance, combining (1) model utility and (2) forgetting score. It is a weighted sum where 'lambda' balances the importance of model utility against forgetting performance. The NoMUS score ranges between 0 (worst) and 1 (best), with higher scores indicating better unlearning.

#### Source Codes

|MUFAC (multi-task)| [Total Experiments](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_Full_Experiments.ipynb) | [Base Models (Original, Retrained)](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_Base_Models.ipynb) |  [Fine-tuning(Standard Fine-tuning, CF-3)](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_FineTuing.ipynb)   | [NegGrad(Standard NegGrad, Advanced NegGrad)](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_NegGrad.ipynb) | [UNSIR](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_UNSIR.ipynb)  | [SCRUB](https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_SCRUB.ipynb) |
|------------------|------------------|--------------------------------------------------|-------|-------------|-----------------|-------|
|**<div align="center">MUCAC (multi-label)</div>**| [<div align="center">**Total Experiments**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_Full_Experiments.ipynb) | [<div align="center">**Base Models (Original, Retrained)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUFAC__Multi_Label_Base_Models.ipynb) |  [<div align="center">**Fine-tuning(Standard Fine-tuning, CF-3)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC__Multi_Label_FineTuning.ipynb)   | [<div align="center">**NegGrad(Standard NegGrad, Advanced NegGrad)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_NegGrad.ipynb) | [**UNSIR**](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_UNSIR.ipynb)  | [**SCRUB**](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_SCRUB.ipynb) |
|**<div align="center">MUCAC (binary-class)</div>**| [<div align="center">**Total Experiments**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_Full_Experiments.ipynb) | [<div align="center">**Base Models (Original, Retrained)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_Base_Models.ipynb) |  [<div align="center">**Fine-tuning(Standard Fine-tuning, CF-3)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_FineTuning.ipynb)   | [<div align="center">**NegGrad(Standard NegGrad, Advanced NegGrad)**</div>](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_NegGrad.ipynb) | [**UNSIR**](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_UNSIR.ipynb)  | [**SCRUB**](https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/binay_classification/Machine_Unlearning_MUCAC_Binary_Cls_SCRUB.ipynb) |


#### Models Performance 
> Detailed performance comparisons of various state-of-the-art unlearning methods applied to our **MUFAC** and **MUCAC** datasets, using a ResNet18 model trained from scratch.

 [Download Original Models for Implementations](https://drive.google.com/drive/folders/1PCj-f2KV7XDiQJEzce0aDRasmjVkb9CN?usp=sharing)
 
1. Overall Performance for <b>Multi-classification</b> on <b>MUFAC</b>

| Metrics                         | Original | Retrained | Fine-tuning | CF-3  | Grad Asc | UNSIR Stage 1 | UNSIR Stage 2 | SCRUB | Advanced Grad Asc |
|---------------------------------|----------|-----------|-------------|-------|----------|---------------|---------------|-------|-------------------|
| <div align="center">Test Acc</div>        | <div align="center">0.5951</div> | <div align="center">0.488</div>  | <div align="center">0.6055</div> | <div align="center">0.5900</div>   | <div align="center">0.4048</div> | <div align="center">0.5893</div>       | <div align="center">0.5925</div>       | <div align="center">0.5984</div> | <div align="center">0.5633</div>          |
| <div align="center">Top-2 Test Acc</div>  | <div align="center">0.8804</div> | <div align="center">0.7667</div> | <div align="center">0.8869</div> | <div align="center">0.8804</div> | <div align="center">0.5932</div> | <div align="center">0.8778</div>       | <div align="center">0.8674</div>       | <div align="center">0.8745</div> | <div align="center">0.8557</div>          |
| <div align="center">MIA</div>             | <div align="center">0.2136</div> | <div align="center">0.0445</div> | <div align="center">0.2129</div> | <div align="center">0.2126</div> | <div align="center">0.0485</div> | <div align="center">0.2089</div>       | <div align="center">0.1990</div>        | <div align="center">0.1415</div> | <div align="center">0.0953</div>          |
| <div align="center">Final Score</div>     | <div align="center">0.5839</div>| <div align="center">0.6995</div>  | <div align="center">0.5898</div>| <div align="center">0.5824</div> | <div align="center">0.6539</div> | <div align="center">0.5857</div>      | <div align="center">0.5972</div>      | <div align="center">0.6577</div> | <div align="center">0.6863</div>         |


<br>

&nbsp;&nbsp;2-1. Overall Performance for <b>Multi-label</b> on <b>MUCAC</b> 
  
| Metrics                         | Original | Retrained | Fine-tuning | CF-3  | Grad Asc | UNSIR Stage 1 | UNSIR Stage 2 | SCRUB | Advanced Grad Asc |
|---------------------------------|----------|-----------|-------------|-------|----------|---------------|---------------|-------|-------------------|
| <div align="center">Test Acc</div>        | <div align="center">0.8852</div>  | <div align="center">0.8135</div>  | <div align="center">0.9147</div> | <div align="center">0.9197</div> | <div align="center">0.4193</div> | <div align="center">0.7087</div>        | <div align="center">0.9220</div>          | <div align="center">0.9073</div>  | <div align="center">0.7607</div>          |
| <div align="center">MIA</div>             | <div align="center">0.0568</div>  | <div align="center">0.0436</div>  | <div align="center">0.0708</div> | <div align="center">0.0685</div> | <div align="center">0.0356</div> | <div align="center">0.0324</div>        | <div align="center">0.0705</div>          | <div align="center">0.0478</div>  | <div align="center">0.0152</div>          |
| <div align="center">Final Score</div>     | <div align="center">0.8858</div>  | <div align="center">0.8631</div> | <div align="center">0.8865</div> | <div align="center">0.8913</div> | <div align="center">0.6740</div> | <div align="center">0.8219</div>       | <div align="center">0.8905</div>          | <div align="center">0.9058</div> | <div align="center">0.8651</div>         |


<br>

&nbsp;&nbsp;2-2. Overall Performance for <b>Binary-classification</b> on <b>MUCAC</b> 
* Male & Female
  
| Metrics                         | Original | Retrained | Fine-tuning | CF-3  | Grad Asc | UNSIR Stage1 | UNSIR Stage2 | SCRUB | Advanced Grad Asc |
|---------------------------------|----------|-----------|-------------|-------|----------|---------------|---------------|-------|-------------------|
| <div align="center">Test Acc</div>        | <div align="center">0.9835</div>  | <div align="center">0.9515</div>  | <div align="center">0.9849</div> | <div align="center">0.9840</div>   | <div align="center">0.1762</div> | <div align="center">0.9481</div>       | <div align="center">0.9845</div>       | <div align="center">0.1762</div> | <div align="center">0.9147</div>          |
| <div align="center">MIA</div>             | <div align="center">0.0306</div>  | <div align="center">0.0154</div>  | <div align="center">0.0281</div> | <div align="center">0.0291</div> | <div align="center">0.1289</div> | <div align="center">0.0638</div>       | <div align="center">0.0481</div>       | <div align="center">0.1329</div> | <div align="center">0.0129</div>          |
| <div align="center">Final Score</div>     | <div align="center">0.9611</div> | <div align="center">0.9603</div> | <div align="center">0.9643</div> | <div align="center">0.9629</div> | <div align="center">0.4592</div> | <div align="center">0.9102</div>      | <div align="center">0.9441</div>      | <div align="center">0.4552</div> | <div align="center">0.9444</div>         |



  * Smiling & Unsmiling

| Metrics                         | Original | Retrained | Fine-tuning | CF-3   | Grad Asc | UNSIR Stage1 | UNSIR Stage2 | SCRUB | Advanced Grad Asc |
|---------------------------------|----------|-----------|-------------|--------|----------|---------------|---------------|-------|-------------------|
| <div align="center">Test Acc</div>        | <div align="center">0.9467</div>  | <div align="center">0.6518</div>   | <div align="center">0.9476</div>  | <div align="center">0.9472</div> | <div align="center">0.5549</div>  | <div align="center">0.8619</div>       | <div align="center">0.9506</div>       | <div align="center">0.5549</div> | <div align="center">0.9423</div>          |
| <div align="center">MIA</div>             | <div align="center">0.0346</div>  | <div align="center">0.0182</div>   | <div align="center">0.0279</div>  | <div align="center">0.0294</div> | <div align="center">0.0366</div>  | <div align="center">0.0416</div>       | <div align="center">0.0271</div>       | <div align="center">0.468</div>  | <div align="center">0.0354</div>          |
| <div align="center">Final Score</div>     | <div align="center">0.9387</div> | <div align="center">0.8077</div>   | <div align="center">0.9459</div>  | <div align="center">0.9442</div> | <div align="center">0.7408</div> | <div align="center">0.8893</div>      | <div align="center">0.9482</div>       | <div align="center">0.3094</div>| <div align="center">0.9357</div>         |


  * Young & Old
    
| Metrics                         | Original | Retrained | Fine-tuning | CF-3   | Grad Asc | UNSIR Stage1 | UNSIR Stage2 | SCRUB | Advanced Grad Asc |
|---------------------------------|----------|-----------|-------------|--------|----------|---------------|---------------|-------|-------------------|
| <div align="center">Test Acc</div>        | <div align="center">0.9089</div>  | <div align="center">0.8271</div>   | <div align="center">0.9147</div>  | <div align="center">0.9118</div> | <div align="center">0.1733</div>  | <div align="center">0.826</div>        | <div align="center">0.9021</div>       | <div align="center">0.8929</div> | <div align="center">0.5573</div>          |
| <div align="center">MIA</div>             | <div align="center">0.0456</div>  | <div align="center">0.0234</div>   | <div align="center">0.0426</div>  | <div align="center">0.0456</div> | <div align="center">0.0513</div>  | <div align="center">0.0229</div>       | <div align="center">0.0493</div>       | <div align="center">0.0428</div>  | <div align="center">0.0139</div>          |
| <div align="center">Final Score</div>     | <div align="center">0.9088</div> | <div align="center">0.8901</div>  | <div align="center">0.9147</div> | <div align="center">0.9103</div> | <div align="center">0.5353</div> | <div align="center">0.8901</div>       | <div align="center">0.9017</div>      | <div align="center">0.9036</div>| <div align="center">0.7647</div>         |


#### Citation
> To be continued...
