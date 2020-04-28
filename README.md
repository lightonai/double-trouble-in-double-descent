# double_trouble
This is the code to reproduce Figure 7 of ["Random projections did it again!"](https://medium.com/p/23992c61ff84/edit) blog post on Medium.

This script recovers the effect of ensembling on the double descent curve using random projections plus a ridge classifier solved via SVD. 

To request access to our cloud and try our optics-based hardware, contact us: https://www.lighton.ai/contact-us/

# to run the script
```
python double_trouble.py  
```

Running `double_trouble.py` outputs a `.npz` file. To plot the results using this file look at the `double_trouble_plot.py` example. 
