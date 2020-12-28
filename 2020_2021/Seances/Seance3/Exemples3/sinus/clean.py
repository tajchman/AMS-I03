
import subprocess, os, shutil

versions = [
    "omp_adaptatif",
    "seq",
    "omp_fine_grain",
    "omp_coarse_grain",
    "omp_tasks"
]

for v in versions:
    print (v)
    for f in ["CMakeCache.txt",
              "Makefile",
              "cmake_install.cmake",
              "CMakeFiles",
              "sinus_" + v + ".exe"]:
        ff = os.path.join("sinus_" + v, f)
        print (ff)
        if os.path.isdir(ff):
            shutil.rmtree(ff)
        elif os.path.isfile(ff):
            os.remove(ff)
