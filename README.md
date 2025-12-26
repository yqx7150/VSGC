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

**Other Related Projects**

SWARM:Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion 
[paper](https://arxiv.org/abs/2501.09935)
[code](https://github.com/yqx7150/SWARM.git)

SWORD:Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction 
[paper](https://ieeexplore.ieee.org/abstract/document/10403850?casa_token=zIecPw7hm1IAAAAA:rVQFq4tyoE0OyPP9E7jTB01LMRpuYf3siYgnYieNN3_R90r207Yn0LZE6G0UKK7ybuGDzN2D5vbAF38)
[code](https://github.com/yqx7150/SWORD.git)
