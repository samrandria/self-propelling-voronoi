Read manual 
=============

# What is this repository?

This is part of the code I wrote during my Master thesis in the laboratory of Prof. Masaki SANO at the University of Tokyo (東京大学).
It is uploaded on Github for _showcasing_ purposes only. Please do not attempt to draw any scientific insight from this code.

## What are we doing here?

I attempted the implementation of a so-called multi-agent "Self-Propelled Voronoi" to simulate the collective migration behaviour
of neural stem cells.

The development of the “Self-Propelled Voronoi” model was motivated
by the need to take into account both cell shapes and cell motility, notably to
propose a simpler mechanism than the Epithelial to Mesenchymal Transition (EMT) to explain collective cell
migration, or to provide a minimal model for 2D systems in confined geometries.

This work is inspired by the following fantastic papers:
- Dapeng Bi, Xingbo Yang, M. Cristina Marchetti, and M. Lisa Manning. _Motility-driven glass and jamming transitions in biological tissues._, **Physical Review X**, 6(2), apr 2016.
- Fabio Giavazzi, Matteo Paoluzzi, Marta Macchi, Dapeng Bi, Giorgio Scita, M. Lisa Manning, Roberto Cerbino, and M. Cristina Marchetti., _Flocking transition in confluent tissues._, **Soft Matter** (2018) 14 (18): 3471–3477

## The science 

In simplified terms:

- We simulate a lattice, tiled by cells seeded randomly, each with a centre of mass, an area and perimeter.

- The cells are subject to interaction with other cells, via a global tissue energy parameter, that relates their areas and their perimeters
    - The cells tend toward a target area $A_0$ and perimeter $P_0$ to minimise their energy.
      
    $$E ≡ \sum_i [KA(A(ri) − A0)^2 + KP (P(ri) − P0)^2]$$
  
    Where E is the tissue energy, KA and KP are two parameters related respectively to the area and perimeter stiffness, A(ri) and P(ri) are the area and perimeter of the cell of center ri, and A0 and P0 are target area and perimeter toward which cells should tend.

- The cells can move, and the equation of motion can be written as $$dr_i/dt = µF_i + v_0 n_i$$
    - With Fi = −∇riE the mechanical force exerted on the i-th cell, E the global tissue energy, µ a motility term, v0 a fixed velocity magnitude, and n_i a polarity vector.
    - The polarity vector is defined as ni = (cos θi, sin θi), and is subject to random rotational
diffusion, modelled as θi(t + ∆t) = θi(t) + eta_i*∆t, with eta_i a white noise term.
    - Inertia is neglected and overdamped motion is assumed.

## Disclaimers

- This code was written in 2018, when I was a Master student _without formal Python training_ or software engineering experience.
- This repository **is not and will not be maintained**.
- Although partly sanitized for uploading purposes, there remains plenty of comments and hacky things in the code. I am sorry about that.
- The decision to write heavy scientific computations in Python was motivated by the readability/high-level aspect of the language, allowing me to prototype faster.
    - Retrospectively, this was probably a terrible idea, and this whole project should really have been C++ (or Julia) (although optimising with Cython did bring considerable improvements).


## Acknowledgements
I am indebted to Prof. Tetsuya HIRAIWA and Dr. Yosuke FUKAI for their help on this code.
All the programming errors and horrors are mine only.


------------------

# How to use the code

The file `flocking_confluent_marchetti.py` contains the main logic of the simulation.
Because the way I coded it was extremely inefficient in native Python, I used Cython optimising static compiler to speed up things.

The compiled version can be called with the `main_spv_script.py`.

`doc/_build/html/index.html` can be opened in your browser to browse the Sphinx documentation for the code. 

I used computing clusters named `PBS`of the University of Tokyo for parameter exploration and running multiple simulations at the same time, via bash scripts.

Some Jupyter notebooks were used for timing functions, quick tests, and visualisation purposes.

------------

# How to use the building scripts on a server

## Creating a directory

Use
`bash make_new_simulation.bash <DIR_NAME>` 
to copy all relevant name from a `SIMULATION_SOURCES_DIR` were the source files are stored.
This bash script calls `local_cython_synchro.bash` to fetch the most up to date files on the server.


## Modify the files as needed

Mostly
* `flocking_confluent_marchetti.py` : to change the initializations or the computations methods
* `parameters_generator.py` : to change the simulation parameters (iteration number, boxsize, K_A, K_P)
* `samplepbs_template.bash` : which contains the `pbs` instructions for the server. Used to control the number of nodes, the queue, etc.
* `masterscript.bash` : to modify the parameter over which to iterate

_Note_ : for now `masterscript.bash` allows iterating only over one parameter.


## Build the Cython files (if using Cython)

Call
`bash build_cython_files.bash`
This will replace the `.py` modules by `.pyx` modules and create `.so` files for Cython.

## Run the simulation 

`bash masterscript.bash`

Will launch the job(s) on PBS. 

-----
-----

# Analyze the results

`bash export_analysis_script.bash <SIMULATION_DIR>`

Will copy the analysis scripts to `<SIMULATION_DIR>`

## Creating a gif to vizualize the dynamics

`python3.6 create_gif.py <NUMBER_OF_ITERATION> <STEP_SIZE_BETWEEN_FRAMES>`

For instance, if there are 4000 iterations and if pictures are logged every 10 iterations, please run the command :

`python3.6 create_gif.py 4000 10`

-----
-----


# Analyzing simulations

A few scripts have been written to study the `.csv` logs produced by the simulations.
They are still very rough and could benefit from a refactoring to :

* Propose a unified csv format and a single function to parse the logs.
* Assemble all the routines in a module and call them with a single script.

For now the analysis scripts are :

* `read_area_perimeter_log.py` : plots the mean area, mean perimeter, as well as the area and perimeter of an individual cell (to specify in the main)
* `read_velocity_log.py` : plots the mean velocity of the system (displacement/time) and the velocity over time of an individual cell (to specify in the main)
* `read_force_tissue_energy_log.py` : plots the evolution of the mean force on the system and the tissue energy over time.
* `MSD.py` : plots the **Mean squared displacement** and attempts to fit a **diffusive** curve and a **ballistic curve** to it. _This function is still not very conclusive_. In addition, plots the **orientational order** and a polar histogram with the **angle orientation** at one iteration (to specify on the main).

To copy the analysis scripts to the `logdirectory` where the `.csv` logs are stored, you may use the `export_analysis_script.bash` script like

```
    bash export_analysis_script.bash TARGET_SIMULATION_DIR
```

_Note_ : if some files use typed Cython syntax, they may need to be compiled before calling the other scripts such as decribed [here](http://docs.cython.org/en/latest/src/reference/compilation.html).
For instance, if the file `voronoi_tesselation.py` uses Python compatibility-breaking syntax, run

```
    mv voronoi_tesselation.py voronoi_tesselation.pyx && cythonize -i voronoi_tesselation.pyx
```
To convert it to a `.pyx` file and to a Cython usable module.

-----
-----


# Optimizing the code

## Using the profiler 

* Modify `build_cython_files.bash` to use the `setup_cython_profiler.py` script.
* \(Optional) Modify the main script default parameters if the script is not called from a job manager
* Build the cython files by running the bash script `build_cython_files.bash`
* Run `python3 script_profiler.py`

_Note_ : you may use `python3 script_profiler.py > profile.log 2>&1 &` to redirect the standard output to a .log file and `&` to put the job in background.

## Analyzing the profiling results

* `snakeviz` is a visual profiling tool that can be invoked by `snakeviz Profile.prof` where `Profile.prof` is the file obtained at the end of the profiling.


-----

# Improvements

## FIXME - TODO - XXX - HACK 

_List here the issues with a FIXME tag_
Using `grep -E "TODO|XXX|HACK" *.py`, you can get a list of every tag repertoried.
A list is available in `todo_tags.txt`.



## From a software quality point of view

The arguments for the values of the different parameters should be processed by the `argparse` module instead of the `sysargv`.
This would allow calling a `--help` flag for instance, or triggering different behaviours such as :
* `--alignment` : to trigger alignment rules
* `--elipseFitting` : to trigger ellipse fitting
* like this.
 


## Ellipse fitting

Although the direction of the ellipse long axis is correct most of the time, the fitting sometimes appear to fail for reasons unknown.
Be careful when using numerical values.
You can turn the plotting of the fitted ellipses and of the direction of their long axes( green quiver) respectively by :
* setting `plot_on=True` in `fit_ellipses.fit_ellipses_to_cells(args, plot_on=True)`
* uncommenting the `plt.quiver` for fitted ellipses angles in `main_spv`



