# for analyzying skip-gram dimensionality using PMI surrogate
python upper_bd_est.py --data text8_pmi --sigma 0.351 --alpha 0.5 --conf_int 0.2
python plot_upper_bd.py --sigma 0.351 --alpha 0.5 --num_runs 1 --data text8_pmi --conf_int 0.05 0.1 0.2 0.5

