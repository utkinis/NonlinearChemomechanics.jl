clear;figure(1);clf;colormap turbo
fs = 8;

set(gcf,'Color','white','Units','centimeters','Position',[15 3 8 10])

mode = 'pure';

simdirs = {['../results/out_spinodal_' mode '_shear_no_coupling']  ...
    ,      ['../results/out_spinodal_' mode '_shear']              ...
    ,      ['../results/out_nucleation_' mode '_shear_no_coupling']...
    ,      ['../results/out_nucleation_' mode '_shear']};

ax_pos = {[0.07 0.6 0.4 0.4]...
    ,     [0.57 0.6 0.4 0.4]...
    ,     [0.07 0.1 0.4 0.4]...
    ,     [0.57 0.1 0.4 0.4]};

its = [80,80,150,150];

itile = 0;
cb_oy = 0.18;

tiledlayout(2,2,'TileSpacing','tight','Padding','compact')

for isim = 1:numel(simdirs)
    simdir = simdirs{isim};
    load([simdir '/params.mat'])
    [rc2,pc2] = ndgrid(rc,pc);
    xc        = rc2.*cos(pc2);
    yc        = rc2.*sin(pc2);
    it        = its(isim);

    load(sprintf('%s/step_%d.mat',simdir,it))

    nexttile
    pcolor(xc,yc,C);shading flat;axis image

    caxis([0.2 0.8])
    axis off;colormap(gca,flip(gray))
    set(gca,'FontSize',fs)
    xlim([0 r0+lr]); ylim([0 r0+lr])
    text(-0.15,1.0,['\bf' char('a'+itile)],'units','normalized','FontSize',fs+1)
    itile = itile + 1;
end

cb = colorbar;cb.Label.String='\itc';cb.Label.FontSize=fs;cb.Layout.Tile='south';

% cb.Position(3) = cb.Position(3)*0.6;
exportgraphics(gcf,['fig_comp_' mode '.png'],'Resolution',300)