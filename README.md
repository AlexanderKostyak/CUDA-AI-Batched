# CUDA-AI-Batched
Visual Studio Port of Ryan Wise's CUDA Neural Network

Software Credit to github user rdw88 at:
https://github.com/rdw88/CUDA-Neural-Network

The pre-2020 repository was not python-driven.  Some of the code might need updated, but there are more than a few activation types and layer structures to assess.  My repository is nearly identical, but includes a VS2022 .sln file and compiles as an executable, there is no python API involved.  Google states that, in many cases, compiled C++ is 100 times faster than python.  Even for front-end configuration of low-level logic, I have a strong distaste for python.

You cannot run the Test.cu or main.cu files in a project at the same time as they both have a main() function.
