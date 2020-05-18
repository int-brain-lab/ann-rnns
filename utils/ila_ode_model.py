import numpy as np
import scipy.linalg

# connectivity matrix (signs only):
C = np.array(
    [[1, 1, 0, 0, -1, -1, 0, 0],
     [1, 1, 1, -1, -1, -1, -1, 1],
     [0, 1, 0, 0, 0, -1, 0, 0],
     [0, 1, 0, 0, 0, -1, 0, 0],
     [-1, -1, 0, 0, 1, 1, 0, 0],
     [-1, -1, -1, 1, 1, 1, 1, -1],
     [0, -1, 0, 0, 0, 1, 0, 0],
     [0, -1, 0, 0, 0, 1, 0, 0]])

to = 0.8    # trial->other trial
ts = 0.15   # trial->self, must be <= 1-to
bo = 0.8    # block->other block
bs = 0.2    # block->self, must be <= 1-bo
bt = 0.1    # block->trial
tb = 0.1    # trial-> block
bac = 0.1   # block->action correct
acb = 0.1   # action correct->block
bai = 0.1   # block->action incorrect
aib = 0.1   # incorrect action->block

# weight strengths (unsigned)
A = np.array(
    [[ts, tb, 0, 0, to, tb, 0, 0],
     [bt, bs, bac, bai, bt, bo, bac, bai],
     [0, acb, 0, 0, 0, acb, 0, 0],
     [0, aib, 0, 0, 0, bai, 0, 0],
     [to, tb, 0, 0, ts, tb, 0, 0],
     [bt, bo, bac, bai, bt, bs, bac, bai],
     [0, acb, 0, 0, 0, acb, 0, 0],
     [0, bai, 0, 0, 0, aib, 0, 0], ])

# full weight matrix:
W = np.multiply(A, C)

# compute top eigvals, assoc right eigenvectors
eigenvals, left_eigenvec, right_eigenvec = scipy.linalg.eig(W, left=True, right=True)

print('Eigenvalues:\n', eigenvals)
print('Dominant signs:\n', np.argwhere(np.real(left_eigenvec[:, 0]) > 0))

# get interesting modes: Only 4 total important modes (eigvectors).
# Two slow modes, with distinct L, R eigenvectors.
# Two fast modes with the same L, R eigenvectors.

# The L eigenvectors look like integrating modes, with states I would
# expect to see in dynamics.
# The R eigenvectors look less interpretable -- are these switching modes?
