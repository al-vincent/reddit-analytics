function DisplayVisualization(Y, C, marker_size, class_names, font_size, fig_width,augmented, show_legend)
% Display 2-D or 3-D scatter-plot
%
%   DisplayVisualization(Y, C)
%   DisplayVisualization(Y, C, marker_size)
%   DisplayVisualization(Y, C, marker_size, class_names, font_size)
%   DisplayVisualization(Y, C, marker_size, class_names, font_size, fig_width)
%   DisplayVisualization(Y, C, marker_size, class_names, font_size, fig_width,augmented)
%   DisplayVisualization(Y, C, marker_size, class_names, font_size, fig_width,augmented, show_legend)
%
% Input:
%   Y : N x 2 or N x 3, 2-D or 3-D coordindates
%   C : N x 1, ground truth class labels, default=ones(N,1)
%   marker_size : the size of dots, by default depending on N
%   class_names : NC x 1 cell strings, text annotations to be put in the
%                median location of each class, default=[]
%   font_size : font size to for displaying class_names, default=10
%   fig_width : the figure width, default=800
%   augmented : the margin added to the display border (in fraction,
%               default=0.1)
%   show_legend : boolean, to show the legend or not, default=false
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


if ~exist('C', 'var') || isempty(C)
    C = ones(size(Y,1),1);
end
if ~exist('augmented', 'var') || isempty(augmented)
    augmented = 0.1;
end
if ~exist('show_legend', 'var') || isempty(show_legend)
    show_legend = false;
end

if ~exist('class_names', 'var') || isempty(class_names)
    class_names = [];
end

if ~exist('font_size', 'var') || isempty(font_size)
    font_size = 10;
end

if ~exist('marker_size', 'var') || isempty(marker_size)
    n = size(Y,1);
    if n<300
        marker_size = 1000;
    elseif n<10000
        marker_size = 100;
    else
        marker_size = 10;
    end
end

if ~exist('fig_width', 'var') || isempty(fig_width)
    fig_width = 800;
end

nc = length(unique(C));

dim = size(Y,2);
if dim~=2 && dim~=3
    error('only 2D or 3D points can be visualized!');
end

figure('Position', [50,50,fig_width,fig_width]);

if dim==2
    axes('position', [0 0 1 1]);
    axis off;
end
    
cmap = distinguishable_colors(nc, repmat(linspace(0,1,2)',1,3));
colormap(cmap);

rng('default');
pm = randperm(size(Y,1));
Y = Y(pm,:);
C = C(pm);

if dim==2
    if show_legend
        gscatter(Y(:,1), Y(:,2), C, [], '.o+*xsdph^v><', marker_size);
    else
        scatter(Y(:,1), Y(:,2), marker_size, C, 'Marker', '.');
    end
else
    scatter3(Y(:,1), Y(:,2), Y(:,3), marker_size, C, 'Marker', '.');
end

set(gca, 'XTick', []);
set(gca, 'YTick', []);

axis equal;

if augmented>0
    xmin = min(Y(:,1));
    xmax = max(Y(:,1));
    ymin = min(Y(:,2));
    ymax = max(Y(:,2));
    xrange = xmax - xmin;
    yrange = ymax - ymin;
    if xrange<yrange
        range = yrange;
    else
        range = xrange;
    end
    centerx = 0.5 * (xmin + xmax);
    centery = 0.5 * (ymin + ymax);
    xmin_aug = centerx - range * 0.5 * (1+augmented);
    xmax_aug = centerx + range * 0.5 * (1+augmented);
    ymin_aug = centery - range * 0.5 * (1+augmented);
    ymax_aug = centery + range * 0.5 * (1+augmented);
    xlim([xmin_aug, xmax_aug]);
    ylim([ymin_aug, ymax_aug]);
end

if ~isempty(class_names)
    nc = numel(unique(C));
    for ci=1:nc
        if nnz(C==ci)>0
            cxy = mean(Y(C==ci,:),1);
            text(cxy(1), cxy(2), class_names{ci}, ...
                'fontsize', font_size, ...
                'color', [0 0 0], ...
                'Interpreter', 'none', ...  % AV change
                'BackgroundColor', 'none', ...%[0.8 0.8 0.8], ...
                'EdgeColor', 'none', ...[0.3 0.3 0.3], ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
        end
    end
end
