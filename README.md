Before running the preprocessing, please adjust parameters for preprocessing specified in the `parameters.py `.Then run the following commands to generated preprocessed data:



Note that you have to cd into the 'preprocess' folder before running the bash script.
`vectorize.py` will generate X.npy and Y.npy for which containes vectorized data and labels. To test on the training set, import your estimator in the `eval.py` and run the script.