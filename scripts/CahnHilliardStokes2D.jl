using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots,LinearAlgebra,Statistics,Printf,ElasticArrays

const USE_GPU = true
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

macro Gdτ_mech()     esc(:( vpdτ_mech*Re_mech*@all(ηsτ)/max_lxy/(r_mech+2.0) )) end
macro Gdτ_mech_av()  esc(:( vpdτ_mech*Re_mech*@av(ηsτ)/max_lxy/(r_mech+2.0)  )) end
macro ρ_dτ_mech_x()  esc(:( Re_mech*@av_xi(ηsτ)/vpdτ_mech/max_lxy            )) end
macro ρ_dτ_mech_y()  esc(:( Re_mech*@av_yi(ηsτ)/vpdτ_mech/max_lxy            )) end
macro ηs_pτ()        esc(:( 1.0/(1.0/@Gdτ_mech() + 1.0/@all(ηs))             )) end
macro ηs_pτ_av()     esc(:( 1.0/(1.0/@Gdτ_mech_av() + 1.0/@av(ηs))           )) end

@parallel function update_potentials!(μ,C,C_o,Pr,τxx,τyy,τxy,Vx,Vy,ηs,ηsτ,vol,bd,γ2,dt,r_mech,Re_mech,vpdτ_mech,max_lxy,dx,dy)
    # chemical potential
    @inn(μ) = @inn(C)*@inn(C)*@inn(C) - @inn(C) + bd*@inn(Pr) - γ2*(@d2_xi(C)/(dx*dx) + @d2_yi(C)/(dy*dy))
    # stresses
    @all(Pr)  = @all(Pr) - r_mech*@Gdτ_mech()*(@d_xa(Vx)/dx + @d_ya(Vy)/dy + vol*(@all(C)-@all(C_o))/dt)
    @all(τxx) = 2.0*@ηs_pτ()*(@d_xa(Vx)/dx + 0.5*@all(τxx)/@Gdτ_mech())
    @all(τyy) = 2.0*@ηs_pτ()*(@d_ya(Vy)/dy + 0.5*@all(τyy)/@Gdτ_mech())
    @all(τxy) = @ηs_pτ_av()*(@d_yi(Vx)/dy + @d_xi(Vy)/dx + @all(τxy)/@Gdτ_mech_av())
    return
end

@parallel function update_fluxes!(qCx,qCy,Vx,Vy,μ,Pr,τxx,τyy,τxy,ηs,ηsτ,dc0,θ_dτ_chem,Re_mech,vpdτ_mech,max_lxy,dx,dy)
    @inn_x(qCx) = (@inn_x(qCx)*θ_dτ_chem - dc0*@d_xa(μ)/dx)/(θ_dτ_chem + 1.0)
    @inn_y(qCy) = (@inn_y(qCy)*θ_dτ_chem - dc0*@d_ya(μ)/dy)/(θ_dτ_chem + 1.0)
    @inn(Vx)    = @inn(Vx) + (-@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy)/@ρ_dτ_mech_x()
    @inn(Vy)    = @inn(Vy) + (-@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx)/@ρ_dτ_mech_y()
    return
end

@parallel function update_concentrations!(C,C_o,qCx,qCy,ηs,ηsτ,dt,ηs0,npow,ηs_rel,ρ_dτ_chem,dx,dy)
    @all(C)   = (@all(C)*ρ_dτ_chem + @all(C_o)/dt - @d_xa(qCx)/dx - @d_ya(qCy)/dy)/(ρ_dτ_chem + 1.0/dt);
    @all(ηs)  = @all(ηs)*(1.0-ηs_rel) + ηs0*10.0^(npow*@all(C))*ηs_rel
    # @inn(ηsτ) = @maxloc(ηs)*(1.0-ηs_rel) + ηs0*10.0^(npow*@maxloc(C))*ηs_rel
    return
end

@parallel_indices (ix,iy) function advect_C!(dC_dt,C,Vx,Vy,dx,dy)
    if (ix<=size(dC_dt, 1) && iy<=size(dC_dt, 2))
         dC_dt[ix,iy] = - max(Vx[ix+1,iy+1],0.0)*(C[ix+1,iy+1]-C[ix  ,iy+1])/dx -
                          min(Vx[ix+2,iy+1],0.0)*(C[ix+2,iy+1]-C[ix+1,iy+1])/dx -
                          max(Vy[ix+1,iy+1],0.0)*(C[ix+1,iy+1]-C[ix+1,iy  ])/dy -
                          min(Vy[ix+1,iy+2],0.0)*(C[ix+1,iy+2]-C[ix+1,iy+1])/dy
    end
    return
end

@parallel function compute_true_τ!(τxx2,τyy2,τxy2,Vx,Vy,ηs,dx,dy)
    @all(τxx2) = 2.0*@all(ηs)*@d_xa(Vx)/dx
    @all(τyy2) = 2.0*@all(ηs)*@d_ya(Vy)/dy
    @all(τxy2) = @av(ηs)*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    return
end

@parallel function compute_residual!(rC,rVx,rVy,rPr,C,C_o,μ,Pr,τxx2,τyy2,τxy2,Vx,Vy,vol,dt,dx,dy)
    @all(rC)  = (@inn(C) - @inn(C_o))/dt - (@d2_xi(μ)/(dx*dx) + @d2_yi(μ)/(dy*dy))
    @all(rVx) = -@d_xi(Pr)/dx + @d_xi(τxx2)/dx + @d_ya(τxy2)/dy
    @all(rVy) = -@d_yi(Pr)/dy + @d_yi(τyy2)/dy + @d_xa(τxy2)/dx
    @all(rPr) = @d_xa(Vx)/dx + @d_ya(Vy)/dy + vol*(@all(C)-@all(C_o))/dt
    return
end

@parallel_indices (ix,iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

# local preconditioning for high viscosity contrasts
@parallel function precond!(ητ,η)
    @inn(ητ) = @maxloc(η)
    return
end

@views function run()
    # dimensionally independent physics
    lx        = 1.0      # domain extent   [m]
    dc0       = 1.0      # mobility factor [m^2/s]
    ηs0       = 1.0      # solid viscosity [Pa*s]
    # scales
    tsc       = lx^2/dc0 # time scale     [s]
    vsc       = dc0/lx   # velocity scale [m/s]
    psc       = ηs0/tsc  # pressure scale [Pa]
    # nondimensional parameters
    ly_lx     = 1.0
    γ_lx      = 1e-2
    ttot_tsc  = 100.0
    Da        = 1e5
    c0        = 0.0
    ca        = 1.0
    vol       = -0*2e-3
    bd_ipsc   = 0*1e-1
    Pe        = 1000.0
    npow      = 0.5
    # dimensionally dependent
    ly        = ly_lx*lx
    γ         = γ_lx*lx
    ttot      = ttot_tsc*tsc
    dt0       = 1.0/Da*tsc
    εbg       = Pe/tsc
    bd        = bd_ipsc/psc
    # numerics
    nx        = 255
    ny        = round(Int,nx*ly_lx)
    maxiter   = 20max(nx,ny)
    ncheck    = ceil(1max(nx,ny))
    nviz      = 1
    εiter     = [1e-4 1e-4 1e-4 1e-4]
    CFL_chem  = 0.05/sqrt(2)
    CFL_mech  = 0.9/sqrt(2)
    ηs_rel    = 1e-1
    # preprocessing
    dx,dy     = lx/nx,ly/ny
    xc,yc     = LinRange(-lx/2+dx/2,lx/2-dx/2,nx  ), LinRange(-ly/2+dy/2,ly/2-dy/2,ny  )
    xv,yv     = LinRange(-lx/2     ,lx/2     ,nx+1), LinRange(-ly/2     ,ly/2     ,ny+1)
    γ2        = γ^2
    c1        = π^4*γ_lx^2 + π^2
    c2        = c1 + Da
    Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
    Re_mech   = 5π
    r_mech    = 0.5
    max_lxy   = max(lx,ly)
    vpdτ_mech = min(dx,dy)*CFL_mech
    vpdτ_chem = min(dx,dy)*CFL_chem
    ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*max_lxy)
    θ_dτ_chem = max_lxy/(Re_chem*vpdτ_chem)
    ## init
    # allocate fields
    C         = @zeros(nx  ,ny  )
    C_o       = @zeros(nx  ,ny  )
    qCx       = @zeros(nx+1,ny  )
    qCy       = @zeros(nx  ,ny+1)
    dC_dt     = @zeros(nx-2,ny-2)
    Pr        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    μ         = @zeros(nx  ,ny  )
    ηs        = @zeros(nx  ,ny  )
    ηsτ       = @zeros(nx  ,ny  )
    τxx2      = @zeros(nx  ,ny  )
    τyy2      = @zeros(nx  ,ny  )
    τxy2      = @zeros(nx-1,ny-1)
    rC        = @zeros(nx-2,ny-2)
    rPr       = @zeros(nx  ,ny  )
    rVx       = @zeros(nx-1,ny-2)
    rVy       = @zeros(nx-2,ny-1)
    # initial conditions
    C        .= c0 .+ ca.*Data.Array(2 .*rand(nx,ny).-1.0)
    @. Vx     = -εbg*(  xv + 0*yc')
    @. Vy     =  εbg*(0*xc +   yv')
    @. ηs     = ηs0*10.0^(npow*C)
    @parallel precond!(ηsτ,ηs)
    @parallel (1:size(C,1), 1:size(C,2)) bc_x!(ηsτ)
    @parallel (1:size(C,1), 1:size(C,2)) bc_y!(ηsτ)
    it = 0; t = 0.0; dt = dt0
    while t < ttot
        # advection
        if Pe > 0.0
            dt_adv    = min(dx/maximum(abs.(Vx)), dy/maximum(abs.(Vy)))/2.1
            dt        = min(dt0,dt_adv)
            if !(dt ≈ dt0)
                Da        = tsc/dt
                c2        = c1 + Da
                Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
                ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*max_lxy)
                θ_dτ_chem = max_lxy/(Re_chem*vpdτ_chem)
            end
            @parallel advect_C!(dC_dt,C,Vx,Vy,dx,dy)
            @. C[2:end-1,2:end-1] += dt*dC_dt
            @parallel (1:size(C,1), 1:size(C,2)) bc_x!(C)
            @parallel (1:size(C,1), 1:size(C,2)) bc_y!(C)
        end
        # save previous time step
        C_o .= C
        # chemomechanics
        iter = 0; errs = 2εiter
        iter_evo = Float64[];errs_evo = ElasticArray{Float64}(undef,length(εiter),0)
        while any(errs .> εiter) && all(isfinite.(errs)) && iter < maxiter
            @parallel update_potentials!(μ,C,C_o,Pr,τxx,τyy,τxy,Vx,Vy,ηs,ηsτ,vol,bd,γ2,dt,r_mech,Re_mech,vpdτ_mech,max_lxy,dx,dy)
            @parallel (1:size(μ,1), 1:size(μ,2)) bc_x!(μ)
            @parallel (1:size(μ,1), 1:size(μ,2)) bc_y!(μ)
            @parallel update_fluxes!(qCx,qCy,Vx,Vy,μ,Pr,τxx,τyy,τxy,ηs,ηsτ,dc0,θ_dτ_chem,Re_mech,vpdτ_mech,max_lxy,dx,dy)
            @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_y!(Vx)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_x!(Vy)
            @parallel update_concentrations!(C,C_o,qCx,qCy,ηs,ηsτ,dt,ηs0,npow,ηs_rel,ρ_dτ_chem,dx,dy)
            @parallel precond!(ηsτ,ηs)
            @parallel (1:size(C,1), 1:size(C,2)) bc_x!(ηsτ)
            @parallel (1:size(C,1), 1:size(C,2)) bc_y!(ηsτ)
            @parallel (1:size(C,1), 1:size(C,2)) bc_x!(C)
            @parallel (1:size(C,1), 1:size(C,2)) bc_y!(C)
            if iter % ncheck == 0
                Pr .-= mean(Pr)
                @parallel compute_true_τ!(τxx2,τyy2,τxy2,Vx,Vy,ηs,dx,dy)
                @parallel compute_residual!(rC,rVx,rVy,rPr,C,C_o,μ,Pr,τxx2,τyy2,τxy2,Vx,Vy,vol,dt,dx,dy)
                errs = [maximum(abs.(rC)), maximum(abs.(rVx))*tsc/vsc, maximum(abs.(rVy))*tsc/vsc, maximum(abs.(rPr[2:end-1,2:end-1]))*tsc]
                append!(errs_evo,errs);push!(iter_evo,iter/max(nx,ny))
                @printf("  iters/nx = %.1f, errC = %1.3e, errVx = %1.3e, errVy = %1.3e, err∇V = %1.3e\n", iter/max(nx,ny), errs...)
            end
            iter += 1
        end
        it += 1; t += dt
        @printf "time step # %d\n" it
        if it % nviz == 0
            fontsz = 12
            opts  = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xaxis=font(fontsz), yaxis=font(fontsz), titlefontsize=fontsz)
            opts2 = (shape = :diamond, xlabel = "niter/nx", ylabel = "err", yscale=:log10, label = ["C" "Vx" "Vy" "∇V"])
            p1 = heatmap(xc[2:end-1],yc[2:end-1], Array(Pr[2:end-1,2:end-1])'; c=:jet, title="Pr", opts...)
            p2 = heatmap(xc,yc, Array(C)';  c=:cool, clims=(-1,1), title="C", opts...)
            p3 = heatmap(xc,yc, Array(μ)';  c=:cool, title="μ", opts...)
            p4 = heatmap(xv,yc, Array(Vx)';  c=:jet, title="Vx", opts...)
            p5 = heatmap(xc,yc, Array(log10.(ηs))';  c=:jet, title="log10(ηs)", opts...)
            p6 = plot(iter_evo, errs_evo'; opts2...)
            display(plot(p1, p2, p3, p4, p5, p6; size=(1200,1500), dpi=200, layout = (3,2)))
        end
        if !all(isfinite.(errs)) error("sim failed") end
    end
    return
end

run()
