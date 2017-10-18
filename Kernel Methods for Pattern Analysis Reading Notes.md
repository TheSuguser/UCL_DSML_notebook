# *Kernel Methods for Pattern Analysis* Reading Notes

Kernel method can be divided into two parts:

1.  A module that performs the mapping into the embedding or **feature space**
2.  A learning algorithm designed to discover **Linear** pattern in that space.

Kernel function: **a computational shortcut**

Advantagements:

1. The algorithm are implemented in such a way that the coordinates of the embedded points are not needed, only their pairwise inner products.
2. The pairwise inner products can be computed efficiently directly from the original data items using a kernel function.

**Regularisation**: For the *ill-conditioned* problem since there is not enough information in the data to precisely specify the solution, an approach that is frequently adopted is to restrict the choice of functions in some way. Such a restriction or bias is referred to as **regularisation**. Perhaps the simplest regulariser is to favour functions that have small norms. For the case of least square regression, this gives the well-known optimisation criterion of **ridge regression.**

