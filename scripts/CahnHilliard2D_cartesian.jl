using ParallelStencil
using Printf,Plots,LinearAlgebra

# @init_parallel_stencil(CUDA, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)

dGdc(c,χ) = log(c) - log(1.0-c) + χ*(1.0-2.0*c)

macro all(A)   esc(:( $A[ix  ,iy  ] )) end
macro inn(A)   esc(:( $A[ix+1,iy+1] )) end
macro inn_x(A) esc(:( $A[ix+1,iy  ] )) end
macro inn_y(A) esc(:( $A[ix  ,iy+1] )) end
macro d2_xi(A) esc(:( $A[ix  ,iy+1] - 2.0*$A[ix+1,iy+1] + $A[ix+2,iy+1] )) end
macro d2_yi(A) esc(:( $A[ix+1,iy  ] - 2.0*$A[ix+1,iy+1] + $A[ix+1,iy+2] )) end
macro d_xa(A)  esc(:( $A[ix+1,iy] - $A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1] - $A[ix,iy] )) end
macro av_xa(A) esc(:( 0.5*($A[ix+1,iy] + $A[ix,iy]) )) end
macro av_ya(A) esc(:( 0.5*($A[ix,iy+1] + $A[ix,iy]) )) end

@parallel_indices (ix,iy) function init_C!(C,dx,dy,r0,lx,ly)
    if checkbounds(Bool,C,ix,iy)
        x,y = -0.5lx + (ix-1)*dx + 0.5dx, -0.5ly + (iy-1)*dy + 0.5dy
        r2   = x*x + y*y
        C[ix,iy] = (r2 < r0*r0) ? 0.99 : 0.01
    end
    return
end

@parallel_indices (ix,iy) function update_potentials!(μ,C,χ,γ2,dx,dy)
    if ix <= size(μ,1) && iy <= size(μ,2)
        if ix == 1
            d2Cdx2 = -C[ix,iy] + C[ix+1,iy]
        elseif ix == size(μ,1)
            d2Cdx2 = C[ix-1,iy] - C[ix,iy]
        else
            d2Cdx2 = C[ix-1,iy] - 2.0*C[ix,iy] + C[ix+1,iy]
        end
        if iy == 1
            d2Cdy2 = -C[ix,iy] + C[ix,iy+1]
        elseif iy == size(μ,2)
            d2Cdy2 = C[ix,iy-1] - C[ix,iy]
        else
            d2Cdy2 = C[ix,iy-1] - 2.0*C[ix,iy] + C[ix,iy+1]
        end
        ΔC      = d2Cdx2/(dx*dx) + d2Cdy2/(dy*dy)
        @all(μ) = dGdc(@all(C),χ) - γ2*ΔC
    end
    return
end

@parallel_indices (ix,iy) function update_fluxes!(qCx,qCy,μ,C,dc0,θ_dτ_chem,dx,dy)
    if ix <= size(μ,1)-1 && iy <= size(μ,2)
        @inn_x(qCx) = (@inn_x(qCx)*θ_dτ_chem - dc0*@av_xa(C)*(1.0-@av_xa(C))*@d_xa(μ)/dx)/(θ_dτ_chem + 1.0)
    end
    if ix <= size(μ,1) && iy <= size(μ,2)-1
        @inn_y(qCy) = (@inn_y(qCy)*θ_dτ_chem - dc0*@av_ya(C)*(1.0-@av_ya(C))*@d_ya(μ)/dy)/(θ_dτ_chem + 1.0)
    end
    return
end

@parallel_indices (ix,iy) function update_concentrations!(C,C_o,qCx,qCy,dt,ρ_dτ_chem,dx,dy)
    if checkbounds(Bool,C,ix,iy)
        @all(C) = (@all(C)*ρ_dτ_chem + @all(C_o)/dt - @d_xa(qCx)/dx - @d_ya(qCy)/dy)/(ρ_dτ_chem + 1.0/dt)
    end
    return
end

@parallel_indices (ix,iy) function compute_residual!(rC,C,C_o,qCx,qCy,dc0,dt,dx,dy)
    if ix <= size(C,1) && iy <= size(C,2)
        @all(rC) = abs( (@all(C) - @all(C_o))/dt + @d_xa(qCx)/dx + @d_ya(qCy)/dy )
    end
    return
end


@parallel_indices (ix,iy) function bc_x!(A)
    A[1  ,iy] = A[2    ,iy]
    A[end,iy] = A[end-1,iy]
    return
end


@parallel_indices (ix,iy) function bc_y!(A)
    A[ix,1  ] = A[ix,2    ]
    A[ix,end] = A[ix,end-1]
    return
end


@views function runme()
    # dimensionally independent
    lx,ly     = 1.0,1.0
    dc0       = 1.0
    # scales
    tsc       = lx^2/dc0
    # nondimensional
    χ         = 2.6
    # dimensionally dependent
    ttot      = 1*tsc
    dt        = 1e-4*tsc
    r0        = 0.2*lx
    # numerics
    nx,ny     = 201,201
    εtol      = 1e-6
    max_iters = 10*nx
    ncheck    = ceil(Int, 0.5*nx)
    nvis      = 10
    CFL_chem  = 0.25
    # preprocessing
    dx,dy     = lx/nx,ly/ny
    xv,yv     = LinRange(0,lx,nx+1),LinRange(0,ly,ny+1)
    xc,yc     = 0.5*(xv[1:end-1]+xv[2:end]),0.5*(yv[1:end-1]+yv[2:end])
    vpdτ_chem = dx*CFL_chem
    γ         = dx
    γ2        = γ^2
    c1        = π^4*(γ/lx)^2 + π^2
    c2        = c1 + lx^2/dc0/dt
    Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
    ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*lx)
    θ_dτ_chem = lx/(Re_chem*vpdτ_chem)
    # init
    C         = @zeros(nx  ,ny  )
    C_o       = @zeros(nx  ,ny  )
    qCx       = @zeros(nx+1,ny  )
    qCy       = @zeros(nx  ,ny+1)
    μ         = @zeros(nx  ,ny  )
    rC        = @zeros(nx  ,ny  )
    # @parallel init_C!(C,dx,dy,r0,lx,ly)
    C .= 0.8.*rand(nx,ny) .+ 0.1
    # action
    tcur = 0.0; it = 1
    while tcur < ttot
        @printf(" - it #%d\n", it)
        C_o .= C
        iter = 1; max_err = 2εtol; err_evo = Float64[]; iter_evo = Float64[]
        while max_err > εtol && iter < max_iters
            @parallel update_potentials!(μ,C,χ,γ2,dx,dy)
            # @parallel bc_x!(μ); @parallel bc_y!(μ)
            @parallel update_fluxes!(qCx,qCy,μ,C,dc0,θ_dτ_chem,dx,dy)
            @parallel update_concentrations!(C,C_o,qCx,qCy,dt,ρ_dτ_chem,dx,dy)
            if iter % ncheck == 0
                @parallel compute_residual!(rC,C,C_o,qCx,qCy,dc0,dt,dx,dy)
                max_err = maximum(rC)*tsc
                @printf(" -- iter #%d, err (C) = %g\n", iter, max_err)
                isfinite(max_err) || error("Simulation failed")
            end
            iter += 1
        end
        if it % nvis == 0
            opts = (c=:turbo,aspect_ratio=1,xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]))
            display(heatmap(xc,yc,Array(C)';opts...))
        end
        tcur += dt; it += 1
    end
    return
end