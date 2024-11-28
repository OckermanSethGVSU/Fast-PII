import numpy as np
import threading
import queue
import time
import os
import sys

class FastPII:
    def __init__(self, filepath, input_dtype, input_shape, window_size, num_workers, block_size, GPU, num_nodes, MIG_IDs, mmap_during_comp, output_path):
        self.filepath = filepath
        self.input_dtype = input_dtype
        self.input_shape = input_shape
        self.window_size = window_size
        self.num_workers = num_workers
        self.block_size = block_size
        self.GPU = GPU
        self.MIG_IDs = MIG_IDs
        self.num_nodes = num_nodes
        self.mmap_during_comp = mmap_during_comp

        self.MPI_NEEDED = (self.num_nodes > 1 or self.MIG_IDs)
        self.producer_done = threading.Event()
        
        

        self.input_array = np.memmap(f"{filepath}", dtype=input_dtype, mode='r', shape=input_shape)
        

        if self.mmap_during_comp:
            self.output_array = np.memmap(f"mmap_during_comp.dat", dtype=input_dtype, mode='w+', shape=input_shape)
        else:
            self.output_array = np.zeros(self.input_shape, dtype=input_dtype)
        self.q = queue.Queue()

        if self.MPI_NEEDED:
            from mpi4py import MPI
            self.MPI = MPI
            self.comm = self.MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.messages = []
            self.output_path = output_path

        if self.GPU or MIG_IDs:
            import cupy as cp
            self.cp = cp

        self.supported_methods = ["sum"]
    
    def gpu_sum(self, message):
        stream = self.cp.cuda.Stream()
        with stream:
            matrix = self.input_array[message[0][0]:message[0][1], message[1][0]:message[1][1]]
            gpuMatrix = self.cp.asarray(matrix)
            """
            Pad the integral image so we don't lose any dimensionality, and then
            compute the sum of each window using the integral image formula
            
            sum of bottom right + sum of top left - sum of button left - sum of top right
            """
            integralImage = self.cp.cumsum(self.cp.cumsum(gpuMatrix, axis=0), axis=1)
            paddedIntegralImage = self.cp.pad(integralImage, ((self.window_size, 0), (self.window_size, 0)), mode='constant')
            windowSum = paddedIntegralImage[self.window_size:, self.window_size:] + paddedIntegralImage[:-self.window_size, :-self.window_size] - \
                        paddedIntegralImage[self.window_size:, :-self.window_size] - paddedIntegralImage[:-self.window_size, self.window_size:]
            windowSum = windowSum[self.window_size - 1:, self.window_size - 1:]

            x, y = windowSum.shape
            self.output_array[message[0][0]: message[0][0] + x, message[1][0]: message[1][0] + y] = windowSum.get()
            if self.MPI_NEEDED:
                self.messages.append((message[0][0], message[0][0] + x, message[1][0], message[1][0] + y))
            else:
                self.q.task_done()
    
    
    def cpu_sum(self, message):
        matrix = self.input_array[message[0][0]:message[0][1], message[1][0]:message[1][1]]
        """
        Pad the integral image so we don't lose any dimensionality, and then
        compute the sum of each window using the integral image formula
        
        sum of bottom right + sum of top left - sum of button left - sum of top right
        """
        
        integralImage = np.cumsum(np.cumsum(matrix, axis=0), axis=1)
        paddedIntegralImage = np.pad(integralImage, ((self.window_size, 0), (self.window_size, 0)), mode='constant')
        windowSum = (paddedIntegralImage[self.window_size:, self.window_size:] + paddedIntegralImage[:-self.window_size, :-self.window_size] - \
                    paddedIntegralImage[self.window_size:, :-self.window_size] - paddedIntegralImage[:-self.window_size, self.window_size:])[self.window_size - 1:, self.window_size - 1:]
        x, y = windowSum.shape
        self.output_array[message[0][0]: message[0][0] + x, message[1][0]: message[1][0] + y] = windowSum
        if self.MPI_NEEDED:
            self.messages.append((message[0][0], message[0][0] + x, message[1][0], message[1][0] + y))
        else:
            self.q.task_done()


    def producer(self):
        rows, cols = self.input_shape
        i, j = 0, 0

        slices = []

        while (i * self.block_size) - (i * (self.window_size - 1)) < rows:
            while (j * self.block_size) - (j * (self.window_size - 1)) < cols:
                startCol = (self.block_size * j) - (j * (self.window_size - 1))
                startRow = (self.block_size * i) - (i * (self.window_size - 1))
                endRow = startRow + self.block_size
                endCol = startCol + self.block_size

                if rows - startRow >= self.window_size and cols - startCol >= self.window_size:
                    """
                    If we are using MPI, we will transmit the slices,
                    otherwise put the slices in the shared queue
                    """
                    if self.MPI_NEEDED or self.MIG_IDs:
                        slices.append([(startRow, endRow), (startCol, endCol), (i, j)])
                    else:
                        self.q.put([(startRow, endRow), (startCol, endCol)])

                j += 1
                if endCol == cols:
                    break
            if endCol == cols and endRow == rows:
                break
            i += 1
            j = 0

        if self.MPI_NEEDED:
            self._mpi_producer(slices)
        
        self.producer_done.set()

    def _mpi_producer(self, slices):
        """
        Send different indices to each MPI worker. Workers will reach out
        as they are free, and then the producer sends the indices that 
        indicate the part of the array they will calculate the sum of. 
        """
        q_size = len(slices)
        for e in slices:
            print("Qsize: ", q_size)
            status = self.MPI.Status()
            idx = self.comm.recv(source=self.MPI.ANY_SOURCE, tag=0, status=status)
            recv_rank = status.Get_source()
            self.comm.send(e, dest=recv_rank, tag=idx)
            q_size -= 1

        """
        Once we have computed the entire array,
        for every node, close every worker
        """
        for node in range(self.size):
            for worker in range(self.num_workers):
                status = self.MPI.Status()
                idx = self.comm.recv(source=self.MPI.ANY_SOURCE, tag=0, status=status)
                recv_rank = int(status.Get_source())
                self.comm.send(-1, dest=recv_rank, tag=idx)

    def consumer(self, worker_number, func, *func_args):
        print("consumer: ", func)
        while not self.producer_done.is_set() or not self.q.empty():
            try:
                message = self.q.get(timeout=1)
                print(message)
                

                if self.GPU:
                    self._gpu_consumer(worker_number, message, func, *func_args)
                else:
                    self._cpu_consumer(message, func, *func_args)
            except queue.Empty:
                pass

    def _gpu_consumer(self, worker_number, message, func, *func_args):

        """
        If we are using MIG, set my visible 
        device equal to 
        """
        if self.MIG_IDs:
            rank = self.comm.Get_rank()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.MIG_IDs[rank]

        else:
            self.cp.cuda.Device(worker_number).use()
        
        if func in self.supported_methods:
            if func == "sum":
                self.gpu_sum(message)
        else:
            func(self, message, *func_args)

        
        

    def _cpu_consumer(self, message, func, *func_args):
        
        if func in self.supported_methods:
            if func == "sum":
                self.cpu_sum(message)
        else:
            func(self, message, *func_args)
        

    def mpi_consumer(self, worker_id, func, *func_args):
        while True:
            """
            Send a message to the producer indicating we are free;
            receive the response that contains what slice of the input
            array this worker will process.
            """
            self.comm.send(worker_id + 1, dest=0, tag=0)
            message = self.comm.recv(source=0, tag=worker_id + 1)

            if message == -1:
                break
            else:
                if self.GPU or self.MIG_IDs:
                    self._gpu_consumer(worker_id, message, func, *func_args)
                else:
                    self._cpu_consumer(message, func, *func_args)

    


    def run(self, func, *func_args):

        
        producer_thread = threading.Thread(target=self.producer)
        
        if self.MPI_NEEDED:
            if self.rank == 0:
                producer_thread.start()
        else:
            producer_thread.start()
       
        consumer_threads = []
        for i in range(self.num_workers):
            if self.MPI_NEEDED:
                consumer_func = self.mpi_consumer
            else:
                consumer_func = self.consumer
            print(i)
            consumer_thread = threading.Thread(target=consumer_func, args=(i,func,*func_args))
            consumer_thread.start()
            consumer_threads.append(consumer_thread)
        if self.MPI_NEEDED:
            if self.rank == 0:
                producer_thread.join()
        else:        
            producer_thread.join()

        for t in consumer_threads:
            t.join()

        if self.MPI_NEEDED:
            rank = self.rank
            comm = self.comm
            size = comm.Get_size()
            # print("rank ", rank)
            """
            Each MPI rank will write their portion of the output to a shared output file.
            The first rank creates the output file and the rest write to it in parallel.
            """
            if rank == 0:
                shared_file_system_array = np.memmap(f"{self.output_path}/output.dat", dtype=self.input_dtype, mode='w+', shape=self.input_shape)
                
                for m in self.messages:
                    # print(m)   
                    shared_file_system_array[m[0]:m[1], m[2]:m[3]] = self.output_array[m[0]:m[1], m[2]:m[3]]            
                shared_file_system_array.flush()
                
                print(f"Rank {rank} saved")
                if rank + 1 != size:
                    comm.bcast(-1, root=0)
                    # comm.send(0, dest=rank + 1, tag=0)

            else:

                print(f"Rank {rank} waiting")
                temp = 0
                comm.bcast(temp, root=0)
                print(f"Rank {rank} started")
                # comm.recv(source=rank - 1, tag=0)
                if rank + 1 != size:
                    comm.send(0, dest=rank + 1, tag=0)
                shared_file_system_array = np.memmap(f"{self.output_path}/output.dat", dtype=self.input_dtype, mode='r+', shape=self.input_shape)
                for m in self.messages:    
                    # print(m)                
                    shared_file_system_array[m[0]:m[1], m[2]:m[3]] = self.output_array[m[0]:m[1], m[2]:m[3]]
                shared_file_system_array.flush()
            

                print(f"Rank {rank} saved")
                
                
                if self.mmap_during_comp:
                    if os.path.exists("mmap_during_comp.dat"):
                        os.remove("mmap_during_comp.dat")


"""
Perform Parallel Integral Image computation (using either threads or GPUs) 
in single-node and multi-node environments. 
Currently only natively supports the sum operator via passing in 
func set to the string "sum"

You can define your own functions to use the parallel framework that will have access to all the instance 
variables it needs to complete computation. See the documentation for more details.

    
Parameters:
-----------
filepath : str
    Path to the file containing the input data.

input_dtype : class
    The type of data contained in the input file

input_shape : tuple of int
    Shape of the input array (rows, columns).

window_size : int
    Size of the window for which to compute the integral image.
    The window shape will be (window_size, window_size)

num_workers : int
    Number of worker threads to use on each node.

block_size : int
    Size of the blocks into which the input array will be divided for processing.

func: str/callable
    Either a string specifying which built in method to use
    or a custom user defined function. 
    The valid string options are currently limited to: "sum" 

GPU : bool, optional (default=False)
    If True, use GPU for computation. Requires CuPy.

num_nodes : int, optional (default=1)
    Number of nodes to use for computation. If greater than 1, MPI will be used.
    This requires mpi4py, mpirun, and a shared file system to write to.

MIG_IDs : list of int, optional (default=None)
    List of MIG (Multi-Instance GPU) IDs to use for GPU computation. 
    This requires mpi4py and the use of mpirun.
    MPI run should spawn the same number of processes
    as the len(MIG_IDs) 
    Currently, only supports single-node.

mmap_during_comp : bool, optional (default=False)
    If True, memory-mapping will be used to store the output during computation to handle large datasets.
    The function will then return a memory-mapped numpy array object that can be used. 

output_path : str, optional (default=./)
    The path to the directory to save the output file (only relevant with mig or multi-node). 
    If this is a multi-node environment, then the path must be to a shared file system. 

Returns:
--------
None
    When we are using multi-node or MIG. The output 
    will be saved in a file called "output.dat"
    at the location specified by output_path. 

numpy.ndarray
    Note: Only in a single node environment

    When mmap_during_comp is False

numpy.memmap
    Note: Only in a single node environment
    
    When mmap_during_comp is true 
    

Examples:
--------

Single Node | 16 Thread | Window_Size = 50 | block size = 1000
    PII_OP("input.dat", np.int32, (50000, 50000), 5, 16, 1000)

Single Node | 4 GPU | Window_Size = 50| block size = 1000
    PII_OP("input.dat", np.int32, (50000, 50000), 50, 4, 1000, GPU=True)

Single Node | MIG | Window_Size = 50 | block size = 1000
    PII_OP("input.dat", np.int32, (50000, 50000), 50, 1, 1000, MIG_IDs=[mig0, mig1, mig2])
    Note: num_workers must be 1

    Requires using mpirun: e.g mpirun -np 3 python3 program.py

    The command follows the general format: mpirun -np <len(MIG_IDs) python3 program.py

Multi-Node | 2 Nodes | 4 GPU per node | Window_Size = 50 | block size = 1000
    Note: Assumes input.dat is on a shared file system
    
    PII_OP("input.dat", np.int32, (50000, 50000), 50, 4, 1000, GPU=True, output_path="path/to/shared/file/system/")

    Requires using mpirun: e.g.  mpirun -np 2 --hosts <host1, host2>  python3 program.py

    The command follows the general format: mpirun -np #num_nodes --hosts <node1, ..., last_node>  python3 program.py

MIG | 1 Node | 4 GPU MIG profiles on the node | Window Size = 50 | Block size = 1000   

    PII_OP("input.dat",  np.int32, (50000, 50000), 50, 1, 1000, MIG_IDS=["mig0","mig1","mig2", "mig3"])
    
    Requires using mpirun: e.g.mpirun -np 4  python3 program.py

    The command follows the general format: mpirun -np #num_mig_ids python3 program.py    
"""

def PII_OP(filepath, input_dtype, input_shape, window_size, num_workers, block_size, func, *func_args, GPU=False, num_nodes=1, MIG_IDs=None, mmap_during_comp=False, output_path="./"):
    pi_op = PIISweep(filepath, input_dtype, input_shape, window_size, num_workers, block_size, GPU, num_nodes, MIG_IDs, mmap_during_comp, output_path)
    pi_op.run(func, *func_args)
    
    if not pi_op.MPI_NEEDED:
       return pi_op.output_array

