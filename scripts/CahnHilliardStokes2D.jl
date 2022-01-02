using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots,Measures,LinearAlgebra,Printf

const USE_GPU = true
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

macro Gdτ()            esc(:( Vpdτ_st*Re_st*@all(ητ)/max_lxy/(r+2.0) )) end
macro Gdτ_av()         esc(:( Vpdτ_st*Re_st*@av(ητ)/max_lxy/(r+2.0)  )) end
macro dτ_ρ_st_av_xi()  esc(:( Vpdτ_st*max_lxy/Re_st/@av_xi(ητ)       )) end
macro dτ_ρ_st_av_yi()  esc(:( Vpdτ_st*max_lxy/Re_st/@av_yi(ητ)       )) end
macro dτ_ρ_ch()        esc(:( Vpdτ_ch*max_lxy/dc0/Re_ch              )) end

@parallel function update_potentials!(μ,C,C_o,Pr,τxx,τyy,τxy,Vx,Vy,η,ητ,vol,δ,r,Δt,δv,Re_st,Re_ch,Vpdτ_st,Vpdτ_ch,max_lxy,dx,dy)
    # chemical potential
    @inn(μ) = @inn(C)^3.0 - @inn(C) - δ*δ*(@d2_xi(C)/(dx*dx) + @d2_yi(C)/(dy*dy)) + vol*@inn(Pr)
    # stresses
    @all(Pr)  = @all(Pr) - r*@Gdτ()*(@d_xa(Vx)/dx + @d_ya(Vy)/dy + δv*(@all(C)-@all(C_o))/Δt)
    @all(τxx) = (@all(τxx) + 2.0*@Gdτ()*@d_xa(Vx)/dx)/(@Gdτ()/@all(η) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@Gdτ()*@d_ya(Vy)/dy)/(@Gdτ()/@all(η) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@Gdτ_av()*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@Gdτ_av()/@av(η) + 1.0)
    return
end

@parallel function update_fluxes!(qCx,qCy,Vx,Vy,μ,Pr,τxx,τyy,τxy,C,η,ητ,dc0,θr_dτ_ch,Vpdτ_st,max_lxy,Re_st,γ,dx,dy)
    @inn_x(qCx) = (@inn_x(qCx) * θr_dτ_ch - dc0 * @d_xa(μ) / dx) / (1.0 + θr_dτ_ch)
    @inn_y(qCy) = (@inn_y(qCy) * θr_dτ_ch - dc0 * @d_ya(μ) / dy) / (1.0 + θr_dτ_ch)
    @inn(Vx)    = @inn(Vx) + (@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pr)/dx)*@dτ_ρ_st_av_xi()
    @inn(Vy)    = @inn(Vy) + (@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pr)/dy)*@dτ_ρ_st_av_yi()
    return
end

@parallel function update_concentrations!(C,C_o,qCx,qCy,η,ητ,Vpdτ_ch,Re_ch,dc0,Δt,max_lxy,η0,γ,dx,dy)
    @all(C) = (@all(C) +  @dτ_ρ_ch() * (@all(C_o) / Δt - (@d_xa(qCx) / dx + @d_ya(qCy) / dy))) / (1.0 + @dτ_ρ_ch()/Δt)
    @all(η) = η0 * 10.0 ^ (γ*@all(C))
    return
end

@parallel_indices (ix,iy) function advect_C!(dC_dt,C,Vx,Vy,dx,dy)
    if (ix<=size(dC_dt, 1) && iy<=size(dC_dt, 2))
         dC_dt[ix,iy] = - (Vx[ix+1,iy+1]>0)*Vx[ix+1,iy+1]*(C[ix+1,iy+1]-C[ix  ,iy+1])/dx -
                          (Vx[ix+2,iy+1]<0)*Vx[ix+2,iy+1]*(C[ix+2,iy+1]-C[ix+1,iy+1])/dx -
                          (Vy[ix+1,iy+1]>0)*Vy[ix+1,iy+1]*(C[ix+1,iy+1]-C[ix+1,iy  ])/dy -
                          (Vy[ix+1,iy+2]<0)*Vy[ix+1,iy+2]*(C[ix+1,iy+2]-C[ix+1,iy+1])/dy
    end
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

function run()
    # dimensionally independent physics
    lx       = 1.0 # m
    dc0      = 1.0 # m^2/s
    η0       = 1.0 # Pa*s
    # scales
    tsc      = lx^2/dc0
    psc      = η0/tsc
    # nondimensional parameters
    ly_lx    = 1.0
    w_lx     = 0.1
    ttot_tsc = 100.0
    c0       = 0.5
    ca       = 0.1
    Δt_tsc   = 1e-7
    γ        = -1.0
    Pe       = 100.0
    # dimensionally dependent
    ly       = ly_lx*lx
    w        = w_lx*lx
    ttot     = ttot_tsc*tsc
    Δt       = Δt_tsc*tsc
    Δt_t     = 1000*Δt_tsc*tsc
    εbg      = 1.0/tsc
    vol      = 0*1e-5/psc
    δv       = -0*1e-7/tsc
    # numerics
    nx       = 511
    ny       = round(Int,nx*ly_lx)
    maxiter  = 50max(nx,ny)
    ncheck   = 100
    nviz     = 10
    εnl      = 1e-4
    CFL_ch   = 0.25/sqrt(2)
    CFL_st   = 0.6/sqrt(2)
    Re_st    = 15π
    r        = 1.0
    # preprocessing
    dx,dy    = lx/nx,ly/ny
    xc,yc    = LinRange(-lx/2,lx/2,nx), LinRange(-ly/2,ly/2,ny)
    max_lxy  = max(lx,ly)
    Vpdτ_st  = min(dx,dy)*CFL_st
    Vpdτ_ch  = min(dx,dy)*CFL_ch
    δ0       = min(dx/lx,dx/ly);
    δ        = δ0*min(lx,ly)
    # array allocations
    Pr       = @zeros(nx  ,ny  )
    dPr      = @zeros(nx  ,ny  )
    dC       = @zeros(nx  ,ny  )
    dC_dt    = @zeros(nx-2,ny-2)
    μ        = @zeros(nx  ,ny  )
    η        = @zeros(nx  ,ny  )
    ητ       = @zeros(nx  ,ny  )
    τxx      = @zeros(nx  ,ny  )
    τyy      = @zeros(nx  ,ny  )
    τxy      = @zeros(nx-1,ny-1)
    dVx      = @zeros(nx+1,ny  )
    dVy      = @zeros(nx  ,ny+1)
    qCx      = @zeros(nx+1,ny  )
    qCy      = @zeros(nx  ,ny+1)
    # initial conditions
    Vx       = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:(nx+1), iy=1:ny] )
    Vy       = Data.Array(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:nx, iy=1:(ny+1)] )
    C        = Data.Array( c0 .+ ca.*(exp.(.-(xc./w).^2 .- (yc'./w).^2) .- 0.1 .* (2.0 .* rand(nx,ny) .- 1.0)) )
    C_o      = copy(C)
    η        .= η0 .* 10.0 .^ (γ .* C)
    @parallel precond!(ητ,η)
    @parallel (1:size(C,1), 1:size(C,2)) bc_x!(ητ)
    @parallel (1:size(C,1), 1:size(C,2)) bc_y!(ητ)
    it       = 0; t = 0.0
    while t < ttot
        # update iter params
        idt      = tsc/Δt
        Re_ch    = real(sqrt(2*π^4*δ0^2-2*π^2+idt+2*sqrt(Complex(π^8*δ0^4-2*π^6*δ0^2+π^4*δ0^2*idt+π^4-π^2*idt))));
        θr_dτ_ch = max_lxy / Vpdτ_ch / Re_ch
        # save previous time step
        C_o     .= C
        iter = 0; err = 2εnl
        while err > εnl && isfinite(err) && iter < maxiter
            if iter % ncheck == 0 dC .= C; dPr .= Pr; dVx .= Vx; dVy .= Vy end
            @parallel update_potentials!(μ,C,C_o,Pr,τxx,τyy,τxy,Vx,Vy,η,ητ,vol,δ,r,Δt,δv,Re_st,Re_ch,Vpdτ_st,Vpdτ_ch,max_lxy,dx,dy)
            @parallel (1:size(μ,1), 1:size(μ,2)) bc_x!(μ)
            @parallel (1:size(μ,1), 1:size(μ,2)) bc_y!(μ)
            @parallel update_fluxes!(qCx,qCy,Vx,Vy,μ,Pr,τxx,τyy,τxy,C,η,ητ,dc0,θr_dτ_ch,Vpdτ_st,max_lxy,Re_st,γ,dx,dy)
            @parallel update_concentrations!(C,C_o,qCx,qCy,η,ητ,Vpdτ_ch,Re_ch,dc0,Δt,max_lxy,η0,γ,dx,dy)
            @parallel precond!(ητ,η)
            @parallel (1:size(C,1), 1:size(C,2)) bc_x!(ητ)
            @parallel (1:size(C,1), 1:size(C,2)) bc_y!(ητ)
            @parallel (1:size(C,1), 1:size(C,2)) bc_x!(C)
            @parallel (1:size(C,1), 1:size(C,2)) bc_y!(C)
            if iter % ncheck == 0
                @printf "  iter # %d: " iter
                dC .-= C; dPr .-= Pr; dVx .-= Vx; dVy .-= Vy
                errs = [
                    maximum(abs.(dC))/maximum(abs.(C)),
                    maximum(abs.(dPr))/maximum(abs.(Pr)),
                    maximum(abs.(dVx))/maximum(abs.(Vx)),
                    maximum(abs.(dVy))/maximum(abs.(Vy))
                ]
                err = maximum(errs)
                @printf "errC = %e, errPr = %e, errVx = %e, errVy = %e\n" errs...
            end
            iter += 1
        end
        @parallel advect_C!(dC_dt,C,Vx,Vy,dx,dy)
        C[2:end-1,2:end-1] .+= Pe .* Δt .* dC_dt
        it += 1; t += Δt
        Δt_adv = min(dx/maximum(abs.(Vx)), dy/maximum(abs.(Vy)))/Pe/2.1
        Δt     = min(Δt*0.999 + Δt_t*0.001,Δt_adv)
        @printf "time step # %d\n" it
        if it % nviz == 0
            opts = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]))
            p1 = heatmap(xc,yc, Array(Pr)'; c=:jet, clims=(-10,10), title="Pr", opts...)
            p2 = heatmap(xc,yc, Array(C)';  c=:cool, clims=(-1,1), title="C", opts...)
            display(plot(p1, p2))
        end
        if !isfinite(err) error("sim failed") end
    end
    return
end

run()
