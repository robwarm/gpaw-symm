from ase.data.g2_1_ref import ex_atomization, atomization
from ase.tasks.io import read_json
from ase.atoms import string2symbols
from ase.units import kcal, mol

data = read_json('molecule-pw.json')

maepbe = 0.0
maeexx = 0.0
print('                 PBE                   EXX')
print('-' * 48)
for name in ex_atomization:
    epberef = atomization[name][2] * kcal / mol
    epbe = (sum(data[atom]['energy'] for atom in string2symbols(name)) -
            data[name]['energy'])
    eexxref = ex_atomization[name][0] * kcal / mol
    eexx = (sum(data[atom]['energy'] + data[atom]['EXX']
                for atom in string2symbols(name)) -
            data[name]['energy'] - data[name]['EXX'])

    maepbe += abs(epbe - epberef) / len(ex_atomization)
    maeexx += abs(eexx - eexxref) / len(ex_atomization)

    print('%-4s %10.3f %10.3f %10.3f %10.3f' %
          (name, epbe, epbe - epberef, eexx, eexx - eexxref))

print('-' * 48)
print('MAE  %10.3f %10.3f %10.3f %10.3f' % (0, maepbe, 0, maeexx))

assert maepbe < 0.025
assert maeexx < 0.05
