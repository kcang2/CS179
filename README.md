# Labs from CS179; Problems from Udacity CS344
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
5) Not sure if this is the proper way, but copy libsndfile\bin\libsndfile-1.dll to Blur(Proj_name)\x64\Debug
6) #define AUDIO_ON 1 at the start of blur.cpp

### How to install OpenCV
1) Download: https://opencv.org/releases.html
2) Install
3) Setup System Environment Variable: Control Panel -> Edit the system environment variables -> Advanced -> Environment Variables -> New System Variable -> Variable name: OPENCV_DIR, Variable value: (Your OpenCV directory)\build\
4) Setup Visual Studio: Configuration(Top): All Configurations (Both debug & release), Platforms(Top): x64
5) Configuration Properties(Left) -> C/C++ -> General -> Additional Include Directories : $(OPENCV_DIR)\include
6) Configuration Properties(Left) -> Linker -> General -> Additional Library Directories : $(OPENCV_DIR)\x64\vc14\lib
7) Configuration Properties(Left) -> Debugging -> Environment : PATH=$(OPENCV_DIR)\x64\vc14\bin
8) Configuration Manager(Top) -> Active solution platform : x64 ; Platform : x64
9) Configuration(Top): Debug ; Configuration Properties(Left) -> Linker -> Input -> Additional Dependencies : opencv_world(XXX)d.lib
10) Configuration(Top): Release ; Configuration Properties(Left) -> Linker -> Input -> Additional Dependencies : opencv_world(XXX).lib
