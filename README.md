# CS179
CUDA Programming

### How to install CUDA:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html.
Note that certain examples in C:\ProgramData\NVIDIA Corporation\CUDA Samples cannot be built successfully.
To run an example (vector addition), just create a new CUDA project.

### How to pass command-line arguments (argc, argv)
argc stands for argument count.
argv refers to the argument vector.
Go to Project Properties -> Configuration Properties -> Debugging -> Command Arguments

### How to install sndfile:
Download here: http://www.mega-nerd.com/libsndfile/#Download.
Go to Project Properties -> Configuration Properties -> VC++ Directories -> Include Directories, paste "libsndfile\include"
Project Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories, paste "libsndfile\lib"
Project Properties -> Configuration Properties -> Linker -> Imput -> Additional Dependencies, paste "libsndfile\lib\libsndfile-1.lib"
#define AUDIO_ON 1 at the start of blur.cpp
