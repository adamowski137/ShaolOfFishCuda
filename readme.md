# Shoal of fish simulation #

This is a project that simulates behavior of a shoal of fish using Boids algortihm.
The algorithm determines velocity vector for every fish based on nearby speciemens position and previous velocity vectors. Every specie has it's own parameters which define their view zone radius, max and min speed, avoidance factor etc. To mix things up a little the fish also avoid the mouse when clicked or held.
The algorithm was implemented in two formats:

- sequential implementation using only CPU.
- parallel implementation using GPU and CUDA technology.

For optimalization purposes both implementations use a grouping grid which helps detect which specimen are in the visible radius.

The image below shows the results for GPU implementaion and 20 000 fish.

![GPU image](/images/GPU_version.png)

As tou can see in the picture the program was able to work in ~300 fps for 20 000 fish. For 100 000 fish it was able to achieve ~30 fps so significantly less, however it was still quiet smooth.
