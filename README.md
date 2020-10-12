# medical_pytorch

To install with Anaconda:
0. (Create a Python3.8 environment, e.g. as conda create -n <env_name> python=3.8, and activate)
2. Install CUDA if not already done and install PyTorch through conda with the command specified by https://pytorch.org/. The tutorial was written using PyTorch 1.6.0. and CUDA10.2., and the command for Linux was at the time 'conda install pytorch torchvision cudatoolkit=10.2 -c pytorch'
3. cd to the project root (where setup.py lives)
4. Execute 'pip install -r requirements.txt'
5. Set paths in mp.paths.py
6. Execute git update-index --assume-unchanged mp/paths.py so that changes in the paths file are not tracked
7. Execute 'pytest' to test the correct installation. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this mark to ignore. The same holds for tests that used datasets which much be previously downloaded.

When using pylint, unnecesary torch and numpy warnings appear. To avoid, include generated-members=numpy.*,torch.* in the .pylintrc file.


Please take into account the style conventions in style_conventions.py
