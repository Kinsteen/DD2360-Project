# DD2360 Final Project
## Ray tracing in a weekend (but parallelized)

### Multi threaded version
Divide the image into smaller tiles (32x32 for example), and launch one thread per core to render a single tile. When a thread finishes, create a new one with another position. Repeat until image is done.

A class "Tile" will describe the result of the work. It would have:
- x and y position
- size_x and size_y (should be the same)
- sample size (maybe implement dynamic sampling?)
- array of computed pixels

A code in the main thread will then combine those arrays into a single pixel array, to avoid concurrent writes to a single array.
We could use mutex, but then the code will need to be rewritten to be ported to CUDA.

### Results
- Single threaded, 5 samples, 500x281: 25 seconds
- Multi threaded, 5 samples, 500x281, 16 threads: 3.5 seconds (7 times faster)