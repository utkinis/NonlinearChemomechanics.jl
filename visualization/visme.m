clear;figure(1);clf;colormap turbo
simdir = '../results/out2';
% load([simdir '/params.mat']);whos
rc2,pc2 = ndgrid()
for it = 1:10
    load(sprintf('%s/step_%d.mat',simdir,it))
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact')
    nexttile

end