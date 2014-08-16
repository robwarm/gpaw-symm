def agts(queue):
    queue.add('hydrogen.py', ncpus=1, walltime=240)
    gs_CO = queue.add('gs_CO.py', ncpus=1, walltime=1000)
    gs_diamond = queue.add('gs_diamond.py', deps=gs_CO, ncpus=1, walltime=100)
    queue.add('rapbe_CO.py', deps=gs_CO, ncpus=16, walltime=2000)
    queue.add('rapbe_diamond.py', deps=gs_diamond, ncpus=16, walltime=1200)
