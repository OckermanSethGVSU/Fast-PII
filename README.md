# Fast-PII
Fast-PII is a library for performing parallel sliding window analysis. Fast-PII allows users to easily use multiple GPUs, CPU threads, or GPU Multi-Instances to process their data in parallel. Fast-PII follows a intuitive master-worker parallelism model, where each worker will ping the master for its next assignment when it is free. Our framework will split your image in such a way that ensures that no resolution is lost due to data chunking and allow the user to define the block size to fit their memory needs. Fast-PII also offers support for both single-node and multi-node environments.


This package primarily uses the integral image method to perform computations. However, it is also designed to be flexible, allowing users to define and pass their own computation methods that can leverage Fast-PII's built-in parallelism.  





###  `PII_OP`
Performs Parallel Integral Image computation (using either threads or GPUs) in single-node, NVIDIA multi-instance GPU (MIG), and multi-node environments. 

Currently only natively supports the sum operator via passing in `func` set to the string "sum"

You can define your own functions to use the parallel framework that will have access to all the instance variables it needs to complete computation.  Below is an example, showing the required function signature: 
```
def custom_function(instance, message, *args):
    
    # Access instance variables
    matrix = instance.input_array[message[0][0]:message[0][1], message[1][0]:message[1][1]]
    output_array = instance.output_array
    # Your custom processing logic here
    # For example, printing the input and output arrays
    print("Input Array:", input_array)
    print("Output Array:", output_array)
```
**Example usage of custom function**
`PII_OP("input.dat", (50000, 50000), 50, 4, 1000, custom_function, *func_args, GPU=True)`

Parameters:
-----------
`filepath: str`
    Path to the file containing the input data. Must be a file compatible with `numpy.memmap`
`input_dtype : class`
    The type of data contained in the input file

`input_shape : (int, int)`
    Shape of the input array in the format of `(rows, columns)`.

`window_size : int`
    Size of the window for which to compute the integral image. The window shape will be `(window_size, window_size)`

`num_workers : int`
    Number of workers to use on each node. If using GPUs, allocate one worker per GPU.

`block_size : int`
    Size of the blocks into which the input array will be divided for processing. blocks will be of shape `(block_size, block_size)`

`func: str/callable`
Either a string specifying which built in method to use (e.g. "sum") or a custom user defined function. 

`GPU : bool`, optional (default=False)
    If True, use GPU for computation. Requires CuPy.

`num_nodes : int`, optional (default=1)
    Number of nodes to use for computation. If greater than 1, MPI will be used. This requires `mpi4py` and the use of `mpirun`. 


`MIG_IDs : [str, ... , str]`, optional (default=None)
    *Note: Currently, only supports single-node.*
    List of MIG (Multi-Instance GPU) IDs to use for GPU computation. 
    This requires `mpi4py` and the use of `mpirun`.
    MPI run should spawn the same number of processes as the `len(MIG_IDs)` 

`mmap_during_comp : bool`, optional (default=False)
    If True, memory-mapping will be used to store the output during computation to handle large matrices. The file will be named `mmmap_during_comp.dat`.

`output_path` : str, optional (default=./)
    The path to the directory to save the output file (only relevant with mig or multi-node). If this is a multi-node environment, then the path must be to a shared file system. 

Returns:
--------
`None`
   When we are using multi-node or MIG. The output will be saved in a file called "output.dat" at the location specified by `output_path`. 

`numpy.ndarray`
    **Note: Only in a single node environment**
    When `mmap_during_comp` is False

`numpy.memmap`
    **Note: Only in a single node environment**
    When mmap_during_comp is True 

Examples:
--------

**Single Node | 16 Thread | Window Size = 50 | Block size = 1000**
`PII_OP("input.dat", np.int32, (50000, 50000), 5, 16, 1000, "sum")`

**Single Node | 4 GPU | Window Size = 50 | Block size = 1000**
`PII_OP("input.dat",  np.int32, (50000, 50000), 50, 4, 1000, "sum", GPU=True)`

**Single Node | MIG | Window Size = 50 | Block size = 1000**
`PII_OP("input.dat", np.int32,(50000, 50000), 50, 1, 1000, "sum", MIG_IDs=[mig0, mig1, mig2])`
    
*Note: num_workers must be 1*

Requires using `mpirun`: e.g `mpirun -np 3 python3 program.py`

The command follows the general format: `mpirun -np <len(MIG_IDs) python3 program.py`

**Multi-Node | 2 Nodes | 4 GPU per node | Window Size = 50 | Block size = 1000**
    
*Note: Assumes input.dat is on a shared file system (networked or otherwise) and writes to that same shared file system.*
    
`PII_OP("input.dat",  np.int32, (50000, 50000), 50, 4, 1000, "sum", GPU=True)`
    

Requires using `mpirun`: e.g.`mpirun -np 2 --hosts <host1, host2>  python3 program.py`

The command follows the general format: `mpirun -np #num_nodes --hosts <node1,..., last_node>  python3 program.py`


**MIG | 1 Node | 4 GPU MIG profiles on the node | Window Size = 50 | Block size = 1000**
`PII_OP("input.dat",  np.int32, (50000, 50000), 50, 1, 1000, "sum", MIG_IDS=["mig0","mig1","mig2", "mig3"])`
    

Requires using `mpirun`: e.g.`mpirun -np 4  python3 program.py`

The command follows the general format: `mpirun -np #num_mig_ids python3 program.py`
