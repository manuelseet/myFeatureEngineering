def pca_cumulative_explained(thispca):  # takes in the fitted pca object
    evr = thispca.explained_variance_ratio_
    cumul = []
    curr = 0
    for i in range(len(evr)):
        if i == 0:
            cumul.append(evr[i])
        else:
            curr = cumul[-1] + evr[i]
            cumul.append(curr)
    return cumul  # returns the array of cumulative explained variance ratios
