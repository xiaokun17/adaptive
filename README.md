# Adaptive SPH
This project is an implementation of **Spatial adaptivity with boundary refinement for smoothed particle hydrodynamics fluid simulation[1]** written in Taichi. 

## Gallery
**Boat sailing 70K particles** \
<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="mark1_ship_withmodel.mp4" type="video/mp4">
</videos>

## Features
**Techniques**
- Adaptive adjustment of fluid particle size based on the distance to specified boundary objects. 
- Adaptivity achieved using splitting and merging scheme in [2]. 
- Fluid simulation using WCSPH [3]. 
- Boundary handling using Semi-analytic boundary handling [4] based on solid SDFs.

## Dependencies
- Python 3.9
- Taichi 0.8.9

## Usage
**How to run**

Execute the following command in the parent directory of the "Taichi_SPH" folder: \
```python -m Taichi_SPH.main```

**Setting up scene**

Please refer to the [Scene configuration guide](documentation\config_guide.md)

## References
[1] Y. Xu, C. Song, X. Wang, X. Ban, J. Wang, Y. Zhang and J. Chang. Spatial adaptivity with boundary refinement for smoothed particle hydrodynamics fluid simulation. Comput Anim Virtual Worlds. 2022;e2136. https://doi.org/10.1002/cav.2136 \
[2] Rene Winchenbach, Hendrik Hochstetter, and Andreas Kolb. Infinite continuous adaptivity for incompressible SPH. ACM Trans. Graph., 36(4), jul 2017. \
[3] M. Becker and . Teschner. Weakly compressible SPH for free surface flows. Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. volume 9, pages 209–217, 01 2007. \
[4] Rene Winchenbach, Rustam Akhunov, and Andreas Kolb. Semi-analytic boundary handling below particle resolution for smoothed particle hydrodynamics. ACM Trans. Graph., 39(6), nov 2020. 
