# CS179
CUDA Programming

### How to install CUDA:
1) Refer to: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html.
2) Note that certain examples in C:\ProgramData\NVIDIA Corporation\CUDA Samples cannot be built successfully.
3) To run an example (vector addition), just create a new CUDA project.

### How to pass command-line arguments (argc, argv)
1) argc stands for argument count.
2) argv refers to the argument vector.
3) Go to Project Properties -> Configuration Properties -> Debugging -> Command Arguments

### How to install sndfile:
1) Download here: http://www.mega-nerd.com/libsndfile/#Download.
2) Go to Project Properties -> Configuration Properties -> VC++ Directories -> Include Directories, paste "libsndfile\include"
3) Project Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories, paste "libsndfile\lib"
4) Project Properties -> Configuration Properties -> Linker -> Imput -> Additional Dependencies, paste "libsndfile\lib\libsndfile-1.lib"
5) #define AUDIO_ON 1 at the start of blur.cpp
