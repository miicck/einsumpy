from einsumpy.test.test_optimize import test_minimize_random_quadratic
import cProfile
import time
import os

start_time = time.time()
dimensions = 1000
cProfile.run(f"test_minimize_random_quadratic(dimension={dimensions}, check_result=False)", "profile.out")
print(f"Time to optimize quadratic in {dimensions} dimensions: {time.time() - start_time}s")
os.system("wget -O gprof2dot.py https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py")
os.system(f"python gprof2dot.py -f pstats profile.out | dot -Tpng -o profile.png")
os.system("xdg-open profile.png")
os.system("rm profile.png profile.out gprof2dot.py")
