% clear the workspace
clear;

% read the input data to a matlab table
merged = readtable('..\\..\\Data\\MergedData_PCA.txt');

% set the index to be 'subreddit'
merged.Properties.RowNames = table2cell(merged(:,'subreddit'));

% remove subreddit from the main table
merged.subreddit = [];

% convert the 'merged' table to an array
X = table2array(merged);

% get the 10NN graph (in place of x2p, for generating similarity matrix)
% knn = fastknn(X, 10); 
% P = double(knn + knn' > 0);
P = sparse(x2p(X));

% calculate the algorithm output values
%Y_sne = sne_p(P);       % stochastic neighbour embedding (SNE)
%Y_ssne = ssne_p(P);     % symmetric SNE
Y_tsne = tsne_p(P);     % t-distributed SNE
%Y_nerv = nerv_p(P);     % Neighbour Retrieval Visualiser
Y_wtsne = wtsne_p(P);   % weighted t-SNE

% C, list of class assignments, is meaningless in the unsupervised case.
% However, if we want to display the subreddit names then we need actual 
% values; otherwise, just create a vector of ones.
C = 1:height(merged); %ones(height(merged));
m_size = 5;
names = merged.Properties.RowNames;
f_size = 5;

% display the output for each method
% output from SNE
% DisplayVisualization(Y_sne,C);
% set(gcf, 'name', 'Reddit data by SNE (optimized by MM)', 'NumberTitle', 'off');
% % output from symmetric SNE
% DisplayVisualization(Y_ssne,C);
% set(gcf, 'name', 'Reddit data by Symmetric SNE (optimized by MM)', 'NumberTitle', 'off');
% % output from t-SNE
DisplayVisualization(Y_tsne,C,m_size, names, f_size);
set(gcf, 'name', 'Reddit data by t-SNE (optimized by MM)', 'NumberTitle', 'off');
% % output from Neighbour Retrieval Visualiser
% DisplayVisualization(Y_nerv,C);
% set(gcf, 'name', 'Reddit data by Neighbour Retrieval Visualiser (optimized by MM)', 'NumberTitle', 'off');
% % output from weighted t-SNE
DisplayVisualization(Y_wtsne, C, m_size, names, f_size);
set(gcf, 'name', 'Reddit data by weighted t-SNE (optimized by MM)', 'NumberTitle', 'off');
print('..\\..\\Images\\weighted_tSNE_500', '-dsvg');