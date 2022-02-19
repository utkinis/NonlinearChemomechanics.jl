clear;figure(1);clf;colormap turbo
set(gcf,'Color','white','Units','centimeters','Position',[5 10 16 0.25*16])
% simdir = '../results/out1';
% simdir = '../results/out_spinodal_pure_shear';
% simdir = '../results/out_spinodal_simple_shear';
% simdir = '../results/out_spinodal_simple_shear_v2';
simdir = '../results/out_nucleation_simple_shear';
% simdir = '../results/out_nucleation_simple_shear2';
% simdir = '../results/out_nucleation_pure_shear';
% simdir = '../results/out_nucleation_pure_shear2';
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
timesteps = 10:100:310;
tiledlayout(1,4,'TileSpacing','tight','Padding','compact')
itile = 0;
for it = timesteps
    load(sprintf('%s/step_%d.mat',simdir,it))
%     Vx = Vr(2:end,:).*cos(pc2) - Vp(:,2:end).*sin(pc2);
%     Vy = Vr(2:end,:).*sin(pc2) + Vp(:,2:end).*cos(pc2);
    nexttile;pcolor(xc,yc,rho(2:end-1,2:end-1));shading flat;axis image;caxis([1.04 1.14]);
    if itile == 0
        box off
        xticks([r0 r0+lr]);xticklabels({'{\itR}_{min}','{\itR}_{max}'})
        yticks([])
        ax = gca;
        ax.XRuler.Axle.Visible = 'off';
        ax.YRuler.Axle.Visible = 'off';
    else
        axis off
    end
    xlim([0 r0+lr]); ylim([0 r0+lr])
    set(gca,'FontSize',8)
    text(-0.15,1.0,['\bf' char('A'+itile)],'FontSize',9,'Units','normalized')
%     drawnow
    itile = itile + 1;
end
cb=colorbar;
cb.Layout.Tile = 'east';
cb.Label.String='\rho/\rho_0';
cb.Label.FontSize = 9;
% cb.Position(3) = cb.Position(3)*0.6;
cb.Ticks = [1.04:0.02:1.14];
exportgraphics(gcf,'fig_rho_vs_time.png','Resolution',300)