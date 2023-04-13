# kernelchallenge

To launch the script, use the following command in terminal:

```
python start.py
```

The code should run in less than a minute. Due to different behaviors of the multiprocessing library between Linux and Windows, the computations may be less efficient on the latter OS. It would then be worth to disable the distributed computation as explained in the start() function of start.py file.


The code is organized as follow:
- /algos: Implementation of regression algorithms for kernels.
- /kernels: Implementation of several kernel functions tried for the challenge.
- /utils: Misc. functions used in some kernels computations.
