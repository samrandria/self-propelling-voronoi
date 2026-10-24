"""
Profiling script
Please invoke it by redirecting the output to a file 
"""

import pstats, cProfile
import flocking_confluent_marchetti
import matplotlib

if __name__=='__main__':
    # Run the simulation script
    cProfile.runctx("flocking_confluent_marchetti.main_spv()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
