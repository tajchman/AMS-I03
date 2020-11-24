
import subprocess

versions = [
    "omp_adaptatif",
    "seq",
    "omp_fine_grain",
    "omp_coarse_grain",
    "omp_tasks"
]

err = 0
for v in versions:
    if err == 0:
        err = subprocess.call(["cmake", "."], cwd = "sinus_" + v)
    if err == 0:
        err = subprocess.call(["make"], cwd = "sinus_" + v)
