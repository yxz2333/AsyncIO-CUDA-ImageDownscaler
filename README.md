# AsyncIO-CUDA-ImageDownscaler

## What
主机多线程异步读取图片，GPU通过CUDA流并发降低图像分辨率(/4)，再传回主机异步输出图片。

旨在实现短时间内实现大量图片分辨率的降低。

## Libraries
OpenCV、CUDA
