# SurfaceNet2D
This is a 2D version of SurfaceNet.

For the original 3D SurfaceNet, please refer: [arXiv](https://arxiv.org/abs/1708.01749), [Code](https://github.com/mjiUST/SurfaceNet)

## Quick Start

```shell
git clone https://github.com/Mi-Dora/SurfaceNet2D.git
cd SurfaceNet2D
pip install -r requirements.txt
cd src
python train.py
```

## Visualization

<center>     <img src="https://github.com/Mi-Dora/SurfaceNet2D/blob/master/images/layout.png"> </center> 

Following two figures are the projections taking from the two cameras above, which should be a 1D matrix. Here, the projection is extended to 64 rows to show clearly.

<center><figure class="half">     <img src="https://github.com/Mi-Dora/SurfaceNet2D/blob/master/images/projection1.png" title="Projection 1" width=400>     <img src="https://github.com/Mi-Dora/SurfaceNet2D/blob/master/images/projection2.png" title="Projection 2" width=400> </figure></centercenter>

Following two figures are the CVC which both have identical dimension with the input image.

<center><figure class="half">     <img src="https://github.com/Mi-Dora/SurfaceNet2D/blob/master/images/cvc1.png" title="CVC 1" width=400>     <img src="https://github.com/Mi-Dora/SurfaceNet2D/blob/master/images/cvc2.png" title="CVC 2" width=400> </figure></centercenter>

All figures above are generated in [projection2d.py](https://github.com/Mi-Dora/SurfaceNet2D/blob/master/src/datasets/projection2d.py).