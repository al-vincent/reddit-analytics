# reddit-analytics
Code for my Data Science MSc dissertation, focussed on applying a data science pipeline to Reddit data.

## What it is
The aim of the work was to investigate whether individual subreddits could be grouped together using clustering approaches, based on a broad set of characteristics (generally, *not* focussed on the text of posts). 

**Data:** the project uses data from https://files.pushshift.io/reddit/, a long-term project that has been collecting posts, comments and other Reddit data for several years.

**Approach:** the project uses the following approach:
- The original data files were manually downloaded
- Features are extracted using MapReduce in Java, and are output to a serise of .txt files
- The data from each of these files is then merged into a single csv using python
- The cleaned, merged data is analysed and a series of approaches applied, including dimensionality reduction (using PCA and t-SNE) and several clustering techniques. PCA is applied using R, and t-SNE is run using both MATLAB and python.
- A series of visualisations are generated to show proximity between subreddits

## Known issues
The project is quite old, and was created when I didn't have much understanding of some basic good practice. For example:
- There's no `requirements.txt` or `environment.yml` file, so it would be very challenging to run the code in its current state
- There are no tests (of any sort)
- The original data is not included in the repo (as a sample), so it may be challenging to reconstruct the results

## Why it's useful
The results and outputs of the work showed that there do seem to be underpinning similarities between subreddits related to the same themes. For example, plotting proxigrams of subreddits showed 'intuitive' clusters of subreddits all linked to sports teams (often in different languages, and different types of sport); gaming; cities and countries; etc. 

## Maintenance
The project is not actively maintained, and has not been updated since ~2018
