# Longitudinal Road Profile Estimation

## Abstract
A new method for non-flat modeling of the longitudinal road profile in front
of a vehicle is proposed. Based on a disparity image the profile is modeled in
V-Disparity space by using uniform cubic B-Splines. The model is tracked over
time using a Kalman Filter. The result is a model of the road profile in camera
coordinates, and based on it an elevation map of the environment relative to
the road surface. The proposed system is real-time capable and is verified by
real test runs.

## Thesis
[Diploma Thesis](./thesis.pdf)

## Code Tags
* C++
* Open CV
* Multicore with openMP
* Vectorization with SSE

## Video
<https://www.youtube.com/watch?v=FKZfKfjN2Kk>

