# import useful libraries 
library(ggplot2)
library(ggfortify)

# set input file
f_in = "C:\\Users\\Al\\OneDrive\\Documents\\RoyalHollowayMSc\\Project\\03_Reddit\\Data\\MergedData.txt"

# read the file into a dataframe
merged = read.csv(f_in)

# set the index to be the 'subreddit' column, and remove 'subreddit' from the df
row.names(merged) = merged$subreddit
merged$subreddit = NULL

# run PCA on the data
pca = prcomp(merged, scale. = T)
# plot the result
#biplot(pca, scale=0, cex=0.5)
autoplot(pca, data=merged, shape=FALSE, label.size=2, label.alpha=0.4, 
         loadings=TRUE, loadings.label=TRUE, loadings.label.size=5)
