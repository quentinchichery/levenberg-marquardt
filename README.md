Levenberg-Marquardt Numpy implementation from scratch.

In the example, I fit an mechanical hardening law (Hollomon) to noisy generated points:

![alt text](images/lm.png "lm")

`scipy.optimize.least_squares` works great when you have the raw function for every points (`y = ax + b`), which is not always the case (ie. iterative stress compute).

Some references:
- https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf
- http://ananth.in/docs/lmtut.pdf