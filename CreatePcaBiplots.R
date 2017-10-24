################################################################################
# Author: A. Vincent
# Description: short script to generate PCA biplots and 
#              scree plots for Reddit data
################################################################################

# import useful libraries 
library(ggplot2)
library(ggfortify)

# set input file
f_in = "C:\\Users\\Al\\OneDrive\\Documents\\RoyalHollowayMSc\\Project\\03_Reddit\\Data\\Output\\MergedData_noPCA_Rescaled.txt"

# read the file into a dataframe
merged = read.csv(f_in)

# set the index to be the 'subreddit' column, and remove 'subreddit' from the df
row.names(merged) = merged$subreddit
merged$subreddit = NULL

# run PCA on the data
pca = prcomp(merged, scale. = T)

# plot the result using a ggplot2-style biplot
autoplot(pca, data=merged, shape=FALSE, label.size=2, label.alpha=0.4, 
         loadings=TRUE, loadings.label=TRUE, loadings.label.size=3.5)

#compute standard deviation of each principal component
std.dev <- pca$sdev
#compute variance
pr.var <- std.dev^2
#check variance of first 5 components
#pr.var[1:5]         # AV: uncomment to print to console

#proportion of variance explained
prop.var.ex <- pr.var/sum(pr.var)
#prop.var.ex[1:5]    # AV: uncomment to print to console

# create a scree plot (using ggplot2 rather than base-R)
max.x = length(prop.var.ex)
df = data.frame(seq(1,max.x), prop.var.ex)
colnames(df) = c('x', 'y')
print(ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point(size=2) + 
          geom_hline(yintercept = 0.01, colour="red", linetype="dashed") + 
          labs(x="Principal Component", y="Proportion of Variance Explained") +
          scale_x_continuous(breaks=c(1:max.x),labels=c(1:max.x),limits=c(1,max.x)))
