"""
This script is called to run the compiled cython files
Call it with the same arguments as flocking_confluent_marchetti.py
"""

import flocking_confluent_marchetti
import matplotlib

if __name__=='__main__':
    # Run the simulation script
    flocking_confluent_marchetti.main_spv()
