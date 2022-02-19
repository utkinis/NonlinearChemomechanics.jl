clear;figure(1);clf;colormap turbo
fs = 8;

set(gcf,'Color','white','Units','centimeters','Position',[15 3 8 9.5])
% simdir = '../results/out1';
% simdir = '../results/out_spinodal_pure_shear';
% simdir = '../results/out_spinodal_simple_shear';
% simdir = '../results/out_spinodal_simple_shear_v2';
% simdir = '../results/out_nucleation_simple_shear';
% simdir = '../results/out_nucleation_simple_shear2';
% simdir = '../results/out_nucleation_pure_shear';
simdir = '../results/out_nucleation_pure_shear';
% simdir = '../results/out_inclusion_pure_shear';
% simdir = '../results/out_inclusion_simple_shear';
% simdir = '../results/out_nucleation_pure_shear_no_coupling';
% simdir = '../results/out_nucleation_simple_shear_no_coupling';
% simdir = '../results/out_spinodal_pure_shear_no_coupling';
% simdir = '../results/out_spinodal_simple_shear_no_coupling';
load([simdir '/params.mat'])
[rc2,pc2] = ndgrid(rc,pc);
xc        = rc2.*cos(pc2);
yc        = rc2.*sin(pc2);
it        = 150;
% tiledlayout(2,2,'TileSpacing','tight','Padding','compact')
itile = 0;

cb_oy = 0.18;

load(sprintf('%s/step_%d.mat',simdir,it))
Vx = Vr(2:end,:).*cos(pc2) - Vp(:,2:end).*sin(pc2);
Vy = Vr(2:end,:).*sin(pc2) + Vp(:,2:end).*cos(pc2);
% tiledlayout(2,2,'TileSpacing','compact','Padding','compact')
% t=nexttile;
ax = axes('Position',[0.07 0.6 0.4 0.4]);
pcolor(ax,xc,yc,etas);shading flat;axis image;caxis([1 100]);axis off
cb = colorbar;cb.Label.String='\eta_s';cb.Label.FontSize=fs;cb.Location='southoutside';
cb.Position(1) = ax.Position(1);
cb.Position(2) = ax.Position(2)-cb_oy;
cb.Position(3) = ax.Position(3);
cb.Position(4) = cb.Position(4)*0.8;
xlim([0 r0+lr]); ylim([0 r0+lr])
set(ax,'ColorScale','log','FontSize',fs)
text(-0.15,1.0,'\bfA','units','normalized','FontSize',fs+1)

ax = axes('Position',[0.57 0.6 0.4 0.4]);
pcolor(xc,yc,sqrt((Vx+100*xc).^2+(Vy-100*yc).^2));shading flat;axis image;axis off
cb = colorbar;cb.Label.String='|{\it\bfv}-{\it\bfv}_{bg}|';cb.Label.FontSize=fs;cb.Location='southoutside';
cb.Position(1) = ax.Position(1);
cb.Position(2) = ax.Position(2)-cb_oy;
cb.Position(3) = ax.Position(3);
cb.Position(4) = cb.Position(4)*0.8;
cb.Ticks = 0:5:25;
caxis([0 25])
set(gca,'FontSize',fs)
xlim([0 r0+lr]); ylim([0 r0+lr])
text(-0.15,1.0,'\bfB','units','normalized','FontSize',fs+1)

ax = axes('Position',[0.07 0.1 0.4 0.4]);
pcolor(xc,yc,Pr/1000);shading flat;axis image;
cb = colorbar;cb.Label.String='{\itp}\times10^{-3}';cb.Label.FontSize=fs;cb.Location='southoutside';
cb.Position(1) = ax.Position(1);
cb.Position(2) = ax.Position(2)-cb_oy;
cb.Position(3) = ax.Position(3);
cb.Position(4) = cb.Position(4)*0.8;
cb.Ticks = 2:4:15;
caxis([2 14]);
hold on
st  = 49;
idx = 1:st:size(xc,1);
idy = 1:st:size(xc,2);
quiver(xc(idx,idy),yc(idx,idy),Vx(idx,idy),Vy(idx,idy),0.5,'filled','Color','k','LineWidth',1)
hold off
axis off;
set(gca,'FontSize',fs)
xlim([0 r0+lr]); ylim([0 r0+lr])
text(-0.15,1.0,'\bfC','units','normalized','FontSize',fs+1)

ax = axes('Position',[0.57 0.1 0.4 0.4]);
pcolor(xc,yc,C);shading flat;axis image
cb = colorbar;cb.Label.String='\itc';cb.Label.FontSize=fs;cb.Location='southoutside';
cb.Position(1) = ax.Position(1);
cb.Position(2) = ax.Position(2)-cb_oy;
cb.Position(3) = ax.Position(3);
cb.Position(4) = cb.Position(4)*0.8;
caxis([0 1])
axis off;colormap(gca,flip(gray))
set(gca,'FontSize',fs)
xlim([0 r0+lr]); ylim([0 r0+lr])
text(-0.15,1.0,'\bfD','units','normalized','FontSize',fs+1)

% cb.Position(3) = cb.Position(3)*0.6;
exportgraphics(gcf,'fig_4_panels_pure.png','Resolution',300)