Instructions:

- pytorch setup.py develop
- mkdir pytorch/proto/build
- cd pytorch/proto/build
- CMAKE_PREFIX_PATH=<...>/pytorch/torch cmake ..
- make

Replace <...> in the above with the start of your path to the pytorch directory.
This should compile and dynamically link hello_world, which can be run from
the build directory.

To build again, delete the pytorch/proto/build directory and repeat the process.