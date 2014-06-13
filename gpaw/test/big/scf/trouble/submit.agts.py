def agts(queue):
    run8a = queue.add('run.py .', ncpus=8, walltime=40 * 60)
    run8b = queue.add('run.py .', ncpus=8, walltime=40 * 60)
    run4a = queue.add('run.py .', ncpus=4, walltime=20 * 60)
    run4b = queue.add('run.py .', ncpus=4, walltime=20 * 60)
    run1 = queue.add('run.py .', ncpus=1, walltime=30)
    analyse = queue.add('analyse.py', ncpus=1, walltime=10,
                        deps=[run8a, run8b, run4a, run4b, run1])

