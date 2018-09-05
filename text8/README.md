Scripts for processing the text8 corpus  
See the `run.sh` file for usage  
After running the scripts, move the signal (.pkl file) and noise (which is a single number printed in STDOUT) to PIP_upper_bound directory for further processing  

File | What's in it?
--- | ---
`PPMI.py` | produces the PPMI matrix (Word2Vec surrogate) from the co-occurrence matrices.  
`log_count.py` | produces the log_count matrix (GloVe surrogate) from the co-occurrence matrices.  
`noise_est.py` | produces the estimated noise standard deviation of the PPMI matrices built on two parts of the splitted dataset.  
`noise_est_logcount.py` | produces the estimated noise standard deviation of the log_count matrices built on two parts of the splitted dataset.  

The .pkl files are obtained empirical spectrums of the PPMI and log count matrices  
