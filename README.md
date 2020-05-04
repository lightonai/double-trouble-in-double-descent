# Random projections did it again!
This is the code to reproduce Figure 7 of ["Random projections did it again!"](https://medium.com/@LightOnIO/random-projections-did-it-again-23992c61ff84) blog post on Medium inspired by [Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime](https://arxiv.org/abs/2003.01054).

This script recovers the effect of ensembling on the double descent curve using random projections plus a ridge classifier solved via SVD. 

## Access to Optical Processing Units

To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/

For researchers, we also a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information. 

## Running the experiment
```
python double_trouble.py  
```

Running `double_trouble.py` outputs a `.npz` file. To plot the results using this file look at the `double_trouble_plot.py` example. 
