def agts(queue):
    gaps = queue.add('gaps.py', ncpus=16, walltime=5 * 60)
    queue.add('submit.agts.py', deps=gaps)
 

if __name__ == '__main__':
    import pickle
    results = pickle.load(open('results.pckl'))
    print results
    pbe, pbe0, pbea, pbe0a, pbeb, pbe0b = results.T
    maea = abs(pbe - pbea).mean()
    maeb = abs(pbe - pbeb).mean()
    mae0a = abs(pbe0 - pbe0a).mean()
    mae0b = abs(pbe0 - pbe0b).mean()
    print maea, maeb, mae0a, mae0b
    assert abs(maea - 0.053) < 0.002
    assert abs(maeb - 0.034) < 0.002
    assert abs(mae0a - 0.092) < 0.002
    assert abs(mae0b - 0.073) < 0.002
