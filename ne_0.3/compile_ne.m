function compile_ne(dim)
% Compile MEX C codes
% dim can be 2 or 3; default dim=2
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end

dimdef = sprintf('-DDIM=%d', dim);


filenames = {...
    'compute_ee_obj_grad_repulsive_barneshut.c', ...
    'compute_linlog_obj_grad_repulsive_barneshut.c', ...
    'compute_mdsks_obj_grad_repulsive_barneshut.c', ...
    'compute_tsne_obj_grad_repulsive_barneshut.c', ...
    'compute_wtsne_obj_grad_repulsive_barneshut.c', ...
    'compute_ssne_obj_grad_repulsive_barneshut.c', ...
    'compute_sne_obj_grad_repulsive_barneshut.c', ...
    'compute_nerv_obj_grad_barneshut.c', ...
    'compute_gammane_repulsive_barneshut.c', ...
    };
nf = length(filenames);

for fi=1:nf
    filename = filenames{fi};
    
    mex('-largeArrayDims', dimdef, filename, 'barnes_hut.c');
end

mex fastknn_mex.cpp

cd CMG_101014
MakeCMG;
cd ../minFunc_2012
mexAll
cd ..
