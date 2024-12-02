# GPU-VPM
*A GPU-based aerodynamics solver*

<img src="images/penn.png" width=200>

University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project

Team:
* [Nadine Adnane](https://www.linkedin.com/in/nadnane/)
* [Dominik Kau](https://www.linkedin.com/in/dominikkau/)
* [Shreyas Singh](https://linkedin.com/in/shreyassinghiitr)

Tested on: 
* **MACHINE 1:** ASUS ROG Zephyrus M16
    * **OS:** Windows 11 
    * **Processor:** 12th Gen Intel(R) Core(TM) i9-12900H, 2500 Mhz, 14 Core(s), 20 Logical Processor(s)     
    * **GPU:** NVIDIA GeForce RTX 3070 Ti Laptop GPU 

* **MACHINE 2:** CETS lab computer
    * **OS:** Windows 10
    * **Processor:** i7-12700 @ 2.10 GHz, 32 GB, 
    * **GPU:** T1000 4096 MB

## Description
A GPU-based Vortex Particle Method (VPM) to simulate airflow in aerodynamics. Positioned between low-fidelity, fast simulations and high-fidelity, resource-intensive CFD methods, this medium-fidelity approach enables realistic aerodynamic modeling without requiring heavy computational resources. We aim to use CUDA (high-performance) or WebGPU (cross-platform accessibility) for the implementation, focusing on high-performance, parallelized vortex particle calculations to make it suitable for early-stage design of aerial vehicles.

## Build Instructions
* Clone this repository.
* Build the project using CMake. We recommend using CMake version 3.30.3.

<img src="images/cmake.png" width=700>

* Open the project using Visual Studio. We recommend Visual Studio 2019.

## Methodology
<img src="images/flowchart.png" width=700>

## Implementation
* We decided to implement this as a CUDA C++ project.
* The VPM solver (for dynamic particles) runs on the GPU, while the VLM solver (for static particles) runs on the CPU.

## Results
<img src="images/particle_field.png" width=500>

* Note - this will be updated once we have more visual output.


## References
* [FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady) (GitHub Repository of CPU implementation)
* [VTK Output](https://github.com/mmorse1217/lean-vtk)
* [FLOWVLM (Vortex Lattice Method)](https://github.com/byuflowlab/FLOWVLM)
* [Stable Vortex Particle Method Formulation for Meshless Large-Eddy Simulation](https://www.nas.nasa.gov/assets/nas/pdf/ams/2022/AMS_20220809_Alvarez.pdf) (Alvarez & Ning, 2023, research paper on reformulated Vortex Particle Method) 
* [NASA presentation slides for introductory concepts ](https://www.nas.nasa.gov/assets/nas/pdf/ams/2022/AMS_20220809_Alvarez.pdf)
* [Reformulated Vortex Particle Method and Meshless Large Eddy Simulation of Multirotor Aircraft](https://scholarsarchive.byu.edu/etd/9589/?utm_source=scholarsarchive.byu.edu%2Fetd%2F9589&utm_medium=PDF&utm_campaign=PDFCoverPages) (Alvarez PhD thesis on reformulated VPM) 
* [Scalable Fast Multipole Accelerated Vortex Methods](https://doi.org/10.1109/IPDPSW.2014.110) (Parallelized VPM) 
[Treecode and fast multipole method for N-body simulation with CUDA](https://arxiv.org/abs/1010.1482) (FMM implementation in CUDA)
