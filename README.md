By default, we abdicate all the preprocessing steps expect for lower-casing the tests. Please run the following commands to generated preprocessed data:
```
python preprocess.py
```

To perporm pre-processing, at variaties of levels, please adjust parameters for preprocessing specified in the `parameters.py `, and run `pipeline()` and `create_data()`
which will generate the np arrays converted from training set and labels.


and run `python test_on_packages.py` to test sklearn packages on the training and test sets.