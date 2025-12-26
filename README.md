# VSGC

**Paper:** Visible Singularitites Guided Multi-scale Correlation Modeling Network for Limited-Angle CT Reconstruction

**Authors:** Yiyang Wen, Liu Shi, Zekun Zhou, WenZhe Shan, Qiegen Liu

Unpublished

Limited-angle computed tomography (LACT) offers
the advantages of reduced radiation dose and shortened scanning
time. However, data incompleteness causes images reconstructed
by traditional algorithms to suffer from blurring along the
ray normal direction and streaking artifacts at the boundaries
of limited data. Currently, most deep learning-based LACT
reconstruction methods focus on multi-domain fusion or the
introduction of generic priors, failing to fully align with the core
imaging characteristics of LACT—such as the directionality of
artifacts and directional loss of structural information—caused
by angular deficiency. Inspired by the theory of visible and
invisible singularities, we propose the Visible Singularities Guided
Multi-Scale Correlation Modeling Network (VSGC) for LACT reconstruction. The design philosophy of VSGC consists
of two core steps: first, extract visible edge features from
LACT images and focus the model’s attention on
these visible edges; second, establish all reasonable correlations
between the visible edge features and other regions of the image
as comprehensively as possible. Additionally, a multi-scale loss
function is employed to constrain the model to converge in
multiple aspects. Finally, qualitative and quantitative validations
are conducted on both simulated and real datasets to verify the
effectiveness and feasibility of the proposed design.

<img width="1038" height="829" alt="898c39640f03a60140419b6cf753258c" src="https://github.com/user-attachments/assets/69107e1e-5586-40ba-9338-8b6d858e3ac5" />

**Diagram of Microlocal Regularity Theorem**

<img width="698" height="601" alt="da83018b417e9241cab6c49c7c374f5c" src="https://github.com/user-attachments/assets/14437471-1a0d-4d14-a9b9-84eb0a9626aa" />

**The network was designed inspired by the Diagram of the microlocal regularity theorem.**

<img width="1385" height="623" alt="2958fed2159a56319f7817cb729bad04" src="https://github.com/user-attachments/assets/b7e4ae40-f3c4-4cc7-8cca-bb129fa778ad" />

**Experimental Results**

![aapm](https://github.com/user-attachments/assets/84d92370-601a-49dc-9034-fbd45a8c967b)

### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/OSDM/blob/main/All-CT.png" >  </div>   
    
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10506793)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/OSDM)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
    
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)

  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  

 * Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-view CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10403850)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SWORD)

  * Dual-Domain Collaborative Diffusion Sampling for Multi-Source Stationary Computed Tomography Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10577271)   [<font size=5>**[Code]**</font>](https://github.com/lizrzr/DCDS-Dual-domain_Collaborative_Diffusion_Sampling)

  * Low-rank Angular Prior Guided Multi-diffusion Model for Few-shot Low-dose CT Reconstruction     
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10776993)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PHD)

  * Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2501.09935)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SWARM)    
                    
  * MSDiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2405.05763)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MSDiff)

  * Ordered-subsets Multi-diffusion Model for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2505.09985)
                          
  * Virtual-mask Informed Prior for Sparse-view Dual-Energy CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2504.07753)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VIP-DECT)

  * Raw_data_generation  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Raw_data_generation)

  * PRO: Projection Domain Synthesis for CT Imaging  [<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2506.13443)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PRO)
       
  * UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2508.17816)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UniSino)

  * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT) 
