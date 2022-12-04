# DD2360 Final Project: Ray tracing in a weekend (but parallelized)

## Multi threaded version
Divide the image into smaller tiles (32x32 for example), and launch one thread per core to render a single tile. When a thread finishes, create a new one with another position. Repeat until image is done.

A class "Tile" will describe the result of the work. It would have:
- x and y position
- size_x and size_y (should be the same)
- sample size (maybe implement dynamic sampling?)
- array of computed pixels

A code in the main thread will then combine those arrays into a single pixel array, to avoid concurrent writes to a single array.
We could use mutex, but then the code will need to be rewritten to be ported to CUDA.

### Final implementation
I had problems with tiles of different sizes with the existing code base, so it will be only square tiles, which is great. The code generates more pixel than the image because of the tiling, so those pixels are cut from the result.

If we could do different tile sizes, it would maybe improve performance a bit.

In the implementation, every tile has a thread, and every thread in launched simultaneously. This may have some performance impact, and could be fixed by launching threads when other are finished. This however is a bit difficult, and I'm not sure if it is necessary for the CUDA porting to come.

The merging code works pretty well, but readability could be improved and performance should also be of concerns. I'm not sure if there's a more optimal way than just cycling through every pixel in a tile, but maybe.

### Results
- Single threaded, 5 samples, 500x281: 25 seconds
- Multi threaded, 5 samples, 500x281, 16 threads: 3.5 seconds (7 times faster)
- CUDA, 5 samples, 500x281: 1.31773 seconds (19 times faster than single thread CPU)