clear;figure(1);clf;colormap turbo
set(gcf,'Color','white')
% simdir = '../results/out1';
% simdir = '../results/out_spinodal_pure_shear';
% simdir = '../results/out_spinodal_simple_shear';
% simdir = '../results/out_spinodal_simple_shear_v2';
% simdir = '../results/out_spinodal_no_shear';
% simdir = '../results/out_nucleation_simple_shear';
% simdir = '../results/out_nucleation_simple_shear2';
% simdir = '../results/out_nucleation_pure_shear';
% simdir = '../results/out_nucleation_pure_shear2';
% simdir = '../results/out_nucleation_no_shear';
% simdir = '../results/out_nucleation_pure_shear_no_coupling';
% simdir = '../results/out_nucleation_simple_shear_no_coupling';
simdir = '../results/out_nucleation_no_shear_no_coupling';
% simdir = '../results/out_spinodal_pure_shear_no_coupling';
% simdir = '../results/out_spinodal_simple_shear_no_coupling';
% simdir = '../results/out_spinodal_no_shear_no_coupling';
% simdir = '../results/out_inclusion_pure_shear';
% simdir = '../results/out_inclusion_simple_shear';
% simdir = '../results/out_inclusion_no_shear';
load([simdir '/params.mat'])
[rc2,pc2] = ndgrid(rc,pc);
xc        = rc2.*cos(pc2);
yc        = rc2.*sin(pc2);
iim       = 0;
for it = 500:500:3000
    load(sprintf('%s/step_%d.mat',simdir,it))
    Vx = Vr(2:end,:).*cos(pc2) - Vp(:,2:end).*sin(pc2);
    Vy = Vr(2:end,:).*sin(pc2) + Vp(:,2:end).*cos(pc2);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact')
    nexttile;pcolor(xc,yc,etas);shading flat;axis image;colorbar;caxis([1 100]);axis off;title('\rm\eta_s')
%     xlim([0 inf]); ylim([0 inf])
    set(gca,'ColorScale','log','FontSize',14)
    nexttile;pcolor(xc,yc,rho(2:end-1,2:end-1));shading flat;axis image;colorbar;caxis([0.99 1.1]);axis off;title('\rm\rho')
    set(gca,'FontSize',14)
%     xlim([0 inf]); ylim([0 inf])
    nexttile;pcolor(xc,yc,Pr);shading flat;axis image;colorbar;
%     if it == 10
%         C_i = C;
%     end
%     nexttile;plot(sqrt(xc(:).^2 + yc(:).^2) ,Pr(:), '.');%shading flat;axis image;colorbar;
%     caxis([-0 10000]);
    hold on
    st  = 45;
    idx = 1:st:size(xc,1);
    idy = 1:st:size(xc,2);
    quiver(xc(idx,idy),yc(idx,idy),Vx(idx,idy),Vy(idx,idy),0.5,'filled','Color','k','LineWidth',1)
    hold off
    axis off;title('\rm\itp')
    set(gca,'FontSize',14)
%     xlim([0 inf]); ylim([0 inf])
    nexttile;
    pcolor(xc,yc,C);shading flat;axis image;
%     cb = colorbar; cb.Label.String = '\itc';
%     nexttile;plot(sqrt(xc(:).^2 + yc(:).^2) ,C_i(:), 'g.', sqrt(xc(:).^2 + yc(:).^2) ,C(:), 'r.');%shading flat;axis image;colorbar;
    caxis([0 1])
    axis off;colormap(gca,flip(gray))
    set(gca,'FontSize',14)
%     xlim([0 inf]); ylim([0 inf])
    sgtitle(it)
%     exportgraphics(gcf,sprintf('anim/step_%04d.png',iim),'Resolution',300)
    iim = iim + 1;
    drawnow
end