# Fiete Lab - IBL ANN-RNNs Project

Code corresponding to the paper [Reverse-engineering
Recurrent Neural Network solutions to a hierarchical inference task for mice](https://www.biorxiv.org/content/10.1101/2020.06.09.142745v2).

### Setup
After cloning the repository, run `pip install -r requirements.txt` to install
the project's dependencies. We used Python 3 and did not test Python 2.

### Running

Our code was written to run on local machine or on a SLURM cluster. There are two main
scripts to run.

1) `train.py`: This will train a RNN with parameters specified inside `utils/params.py`.
During training, two types of objects will be written to disk inside a newly created
`runs/` directory. The first is a TensorBoard
file which logs basic information (e.g. loss, average number of RNN steps per trial,
average fraction of correct actions per trial). The second are model checkpoints through
the training process.

2) `analyze.py`: This will take a trained RNN and generate all the plots contained in the
paper. You'll need to specify the train run id (e.g. `rnn, block_side_probs=0.8, max_stim_strength=2.5_2020-06-18 11:14:27.427969`)
inside `analyze.py`. The plots 
will be written to disk inside that run directory, in a newly created directory named 
`analyze`. A PDF containing all the images will also be generated, in case you want to 
send them all to someone simultaneously.

If you want to run on a SLURM cluster, use the `ann-rnn.sh` bash script.

### Notes

- Some of the variable names are inconsistent throughout the repo
 because our understanding evolved as the project progressed. Sadly, I haven't
 had time to rename variables for reader clarity.

### Questions? Concerns? Collaborations?

We'd be delighted to hear from you. Email Rylan Schaeffer at rylanschaeffer@g.harvard.edu 
and cc Ila Fiete at fiete@mit.edu.
