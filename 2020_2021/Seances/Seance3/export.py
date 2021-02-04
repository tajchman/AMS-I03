#! /usr/bin/env python

import os, glob, shutil

for a in glob.glob(os.path.join('TP1*.gz')) \
        + glob.glob(os.path.join('TP1*.zip')) \
        + glob.glob('Exemples3.*'):
  os.remove(a)
  
for d in ['OpenMP_CoarseGrain', 'OpenMP_FineGrain']:
  base = 'TP1_' + d + '_incomplet'

  shutil.make_archive(base, 'zip', d, 'TP1')
  shutil.make_archive(base, 'gztar', d, 'TP1')

for d in ['Exemples3']:
  base = d
  shutil.make_archive(base, 'zip', d)
  shutil.make_archive(base, 'gztar', d)

