# myFeatureEngineering
Built custom feature engineering functions [NUS GradDipSA - Nov 2021]


1. `pca_plot(data, n_comp, threshold)`

  PCA scree plot function to illustrate the explained variance ratio of individual principal components, as well as the cumulative explained variance. 
  Takes in as input the feature value dataset, the number of components to select, and the threshold (0.0 - 1.0) 
  
 <img src="https://github.com/manuelseet/myFeatureEngineering/blob/main/myPCA_plot.png"  alt="PCA scree plot"/> 
 
 
2. `pca_cumulative_explained(thispca)`

 Takes in as input the fitted PCA object. 
 Returns the array of cumulative explained variance, with each additional principal component 
