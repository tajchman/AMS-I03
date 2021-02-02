
for v in ['Cuda', 'Seq']:
  s = 0.0
  p = []
  with open('res_' + v+ '/out', 'r') as f:
    for l in f:
      l = l.replace('\0', '').replace('\r', '').replace('\n', '').replace('\x00', '')
      l = l.strip()
      if len(l) > 0:
        try:
          x = float(l)
          s = s + x
          p.append(x)
        except:
          pass
  print(v)
  p.sort()
  with open('res_' + v+ '/out_sorted', 'w') as f:
    for x in p:
      f.write(str(x) + '\n')
  print(s)
