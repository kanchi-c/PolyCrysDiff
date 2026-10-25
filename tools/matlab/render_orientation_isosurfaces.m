function render_orientation_isosurfaces(matFile, outputDir)
% Render orientation classes 1--8 using the same isosurface convention as Fig. 7.
loaded = load(matFile, 'model');
model = double(loaded.model);
model = model(1:64, 1:64, 1:64);

cmap = [0.12156863 0.46666667 0.70588235
        0.68235294 0.78039216 0.90980392
        1.00000000 0.49803922 0.05490196
        1.00000000 0.73333333 0.47058824
        0.17254902 0.62745098 0.17254902
        0.59607843 0.87450980 0.54117647
        0.83921569 0.15294118 0.15686275
        1.00000000 0.59607843 0.58823529];

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for orientation = 1:8
    fprintf('Rendering orientation %d/8\n', orientation);
    fig = figure('Visible', 'off', 'Color', 'white', ...
                 'Position', [100 100 900 900]);
    ax = axes(fig, 'Position', [0 0 1 1]);
    hold(ax, 'on');
    mask = (model == orientation);
    [faces, vertices] = isosurface(mask, 0.5);
    if ~isempty(faces)
        patch(ax, 'Faces', faces, 'Vertices', vertices, ...
              'FaceColor', cmap(orientation, :), 'EdgeColor', 'none');
    end
    draw_cube_edges(ax, 64);
    axis(ax, 'equal');
    axis(ax, [0 64 0 64 0 64]);
    axis(ax, 'off');
    view(ax, 135, 25);
    camlight(ax, 'headlight');
    lighting(ax, 'gouraud');

    stem = fullfile(outputDir, sprintf('orientation_%d', orientation));
    set(fig, 'PaperUnits', 'inches', ...
             'PaperPosition', [0 0 9 9], ...
             'PaperSize', [9 9], ...
             'InvertHardcopy', 'off');
    print(fig, [stem '.png'], '-dpng', '-r500', '-loose');
    print(fig, [stem '.eps'], '-depsc', '-painters', '-loose');
    close(fig);
end
end


function draw_cube_edges(ax, edgeLength)
vertices = [0 0 0; edgeLength 0 0; edgeLength edgeLength 0; 0 edgeLength 0; ...
            0 0 edgeLength; edgeLength 0 edgeLength; ...
            edgeLength edgeLength edgeLength; 0 edgeLength edgeLength];
edges = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; ...
         1 5; 2 6; 3 7; 4 8];
for index = 1:size(edges, 1)
    points = vertices(edges(index, :), :);
    plot3(ax, points(:, 1), points(:, 2), points(:, 3), ...
          'Color', [0 0 0], 'LineWidth', 0.5);
end
end
