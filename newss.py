from featurewiz import featurewiz

outputs = featurewiz(dataset.iloc[:totalRows,:], collist, corr_limit=0.9, verbose=1, dask_xgboost_flag=False)