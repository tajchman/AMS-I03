
import subprocess, os, argparse

versions = [
    "seq",
    "omp_fine_grain",
    "omp_coarse_grain",
    "omp_adaptatif",
    "omp_tasks"
]

parser = argparse.ArgumentParser()
parser.add_argument('threads', type=int, default=3)
args = parser.parse_args()

err = 0
for v in versions:
    vv = "sinus_" + v
    print("\n___________________________") 
    print(vv) 
    out = subprocess.check_output([os.path.join(vv, vv + ".exe"),
                                   "threads=" + str(args.threads)])
    out = out.decode('utf8')
    r0 = out.find('temps calcul : ')
    r1 = out.find(' s', r0)
    print(out[r0:r1+2], end='')
    t = float(out[r0 + 15:r1])
    if v == "seq":
        t0 = t
        print('\n')
    else:
        print (" speedup = ", int(100*t0/t)/100)
