import subprocess
import os
from tqdm import tqdm

os.makedirs('figures', exist_ok=True)
list_files = os.listdir('data')
for entry in tqdm(os.scandir('data'), total=len(list_files)):
  name = entry.name.split('.')[0]
  path1 = 'figures/' + 'disp_' + name + '.jpg'
  path2 = 'figures/' + 'model_' + name + '.jpg'
  command = '../../python/plot_inversion.py --data {:s} --plot_model --out {:s} --no_show'.format(name, path2)
  subprocess.call(command, shell=True)
  command = '../../python/plot_inversion.py --data {:s} --plot_disp --out {:s} --no_show'.format(name, path1)
  subprocess.call(command, shell=True)
