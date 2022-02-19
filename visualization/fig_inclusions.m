clear;figure(1);clf;colormap turbo

set(gcf,'Color','white','Units','centimeters','Position',[15 3 8 10])

simdirs = {'../results/out_inclusion_no_shear'...
    ,          '../results/out_inclusion_pure_shear'...
    ,          '../results/out_inclusion_simple_shear'};

tiledlayout(3,2,'TileSpacing','none','Padding','compact')

its = [300,1500,1500];

fs = 8;
itile = 0;

for isim = 1:numel(simdirs)
    simdir = simdirs{isim};
    load([simdir '/params.mat'])
    [rc2,pc2] = ndgrid(rc,pc);
    xc        = rc2.*cos(pc2);
    yc        = rc2.*sin(pc2);
    it        = its(isim);

    load(sprintf('%s/step_%d.mat',simdir,it))

    Vx = Vr(2:end,:).*cos(pc2) - Vp(:,2:end).*sin(pc2);
    Vy = Vr(2:end,:).*sin(pc2) + Vp(:,2:end).*cos(pc2);

    nexttile
    pcolor(xc,yc,Pr/1000);shading flat;axis image
    %     caxis([4000 6000])
    axis off;colormap(gca,'turbo')
    set(gca,'FontSize',fs)
    xlim([-(r0+lr) 0]); ylim([0 r0+lr])

    %     itile = itile + 1;

    %     if itile == 5
    cb = colorbar;cb.Label.String='{\itp}\times10^{-3}';cb.Label.FontSize=fs;cb.Location='westoutside';
    %     end

    nexttile
    pcolor(xc,yc,C);shading flat;axis image
    caxis([0 1])
    if itile > 0
        hold on
        st  = 25;
        idx = 1:st:size(xc,1);
        idy = 1:st:size(xc,2);
        quiver(xc(idx,idy),yc(idx,idy),Vx(idx,idy),Vy(idx,idy),0.5,'filled','Color','k','LineWidth',1)
        hold off
    end
    axis off;colormap(gca,flip(gray))
    set(gca,'FontSize',fs)
    xlim([0 r0+lr]); ylim([0 r0+lr])
    %     text(-0.15,1.0,['\bf' char('A'+itile)],'units','normalized','FontSize',fs+1)
    text(0.9,0.9,['\bf' char('A'+itile)],'units','normalized','FontSize',fs+1)

    itile = itile + 1;

    %     if itile == 6
    %         cb = colorbar;cb.Label.String='\itc';cb.Label.FontSize=fs;cb.Location='southoutside';
    %     end
end

% cb.Position(3) = cb.Position(3)*0.6;
exportgraphics(gcf,'fig_inclusions.png','Resolution',300)