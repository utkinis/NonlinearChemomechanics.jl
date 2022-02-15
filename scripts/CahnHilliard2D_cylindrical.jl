using ParallelStencil
using Printf,Plots,LinearAlgebra

@init_parallel_stencil(CUDA, Float64, 2)
# @init_parallel_stencil(Threads, Float64, 2)

dGdc(c,χ) = log(c) - log(1.0-c) + χ*(1.0-2.0*c)
# dGdc(c,χ) = c

macro all(A)   esc(:( $A[ir  ,iϕ  ] )) end
macro inn(A)   esc(:( $A[ir+1,iϕ+1] )) end
macro inn_r(A) esc(:( $A[ir+1,iϕ  ] )) end
macro inn_ϕ(A) esc(:( $A[ir  ,iϕ+1] )) end
macro d_ra(A)  esc(:( $A[ir+1,iϕ] - $A[ir,iϕ] )) end
macro d_ϕa(A)  esc(:( $A[ir,iϕ+1] - $A[ir,iϕ] )) end
macro av_ra(A) esc(:( 0.5*($A[ir+1,iϕ] + $A[ir,iϕ]) )) end
macro av_ϕa(A) esc(:( 0.5*($A[ir,iϕ+1] + $A[ir,iϕ]) )) end

@parallel_indices (ir,iϕ) function init_C!(C,dr,dϕ,r0,lr)
    if checkbounds(Bool,C,ir,iϕ)
        r = r0 + (ir-1)*dr + 0.5dr
        C[ir,iϕ] = (r < 0.5lr) ? 0.999 : 0.001
    end
    return
end

@parallel_indices (ir,iϕ) function update_potentials!(μ,C,χ,γ2,r0,dr,dϕ)
    if ir <= size(μ,1) && iϕ <= size(μ,2)
        rc      = r0+(ir-1)*dr+0.5*dr
        rvn,rvs = rc+0.5*dr,rc-0.5*dr
        rc2     = rc*rc
        if ir == 1
            d2Cdr2 = rvn*(C[ir+1,iϕ] - C[ir,iϕ])/rc
        elseif ir == size(μ,1)
            d2Cdr2 = - rvs*(C[ir,iϕ] - C[ir-1,iϕ])/rc
        else
            d2Cdr2 = (rvn*(C[ir+1,iϕ] - C[ir,iϕ]) - rvs*(C[ir,iϕ] - C[ir-1,iϕ]))/rc
        end
        if iϕ == 1
            d2Cdϕ2 = (-C[ir,iϕ] + C[ir,iϕ+1])/rc2
        elseif iϕ == size(μ,2)
            d2Cdϕ2 = (C[ir,iϕ-1] - C[ir,iϕ])/rc2
        else
            d2Cdϕ2 = (C[ir,iϕ-1] - 2.0*C[ir,iϕ] + C[ir,iϕ+1])/rc2
        end
        ΔC      = d2Cdr2/(dr*dr) + d2Cdϕ2/(dϕ*dϕ)
        @all(μ) = dGdc(@all(C),χ) - γ2*ΔC
    end

    return
end

@parallel_indices (ir,iϕ) function update_fluxes!(qCr,qCϕ,μ,C,dc0,θ_dτ_chem,r0,dr,dϕ)
    if ir <= size(μ,1)-1 && iϕ <= size(μ,2)
        @inn_r(qCr) = (@inn_r(qCr)*θ_dτ_chem - dc0*@av_ra(C)*(1.0-@av_ra(C))*@d_ra(μ)/dr)/(θ_dτ_chem + 1.0)
    end
    if ir <= size(μ,1) && iϕ <= size(μ,2)-1
        rc = r0+(ir-1)*dr+0.5*dr
        @inn_ϕ(qCϕ) = (@inn_ϕ(qCϕ)*θ_dτ_chem - dc0*@av_ϕa(C)*(1.0-@av_ϕa(C))/rc*@d_ϕa(μ)/dϕ)/(θ_dτ_chem + 1.0)
    end
    return
end

@parallel_indices (ir,iϕ) function update_concentrations!(C,C_o,qCr,qCϕ,dt,ρ_dτ_chem,r0,dr,dϕ)
    if checkbounds(Bool,C,ir,iϕ)
        rc      = r0+(ir-1)*dr+0.5*dr
        rvn,rvs = rc+0.5*dr,rc-0.5*dr
        @all(C) = (@all(C)*ρ_dτ_chem + @all(C_o)/dt - (rvn*qCr[ir+1,iϕ]-rvs*qCr[ir,iϕ])/dr/rc - @d_ϕa(qCϕ)/dϕ/rc)/(ρ_dτ_chem + 1.0/dt)
    end
    return
end

@parallel_indices (ir,iϕ) function compute_residual!(rC,C,C_o,qCr,qCϕ,dt,r0,dr,dϕ)
    if ir <= size(C,1) && iϕ <= size(C,2)
        rc      = r0+(ir-1)*dr+0.5*dr
        rvn,rvs = rc+0.5*dr,rc-0.5*dr
        @all(rC) = abs( (@all(C) - @all(C_o))/dt + (rvn*qCr[ir+1,iϕ]-rvs*qCr[ir,iϕ])/dr/rc + @d_ϕa(qCϕ)/dϕ/rc )
    end
    return
end

@parallel_indices (ir,iϕ) function bc_r!(A)
    A[1  ,iϕ] = A[2    ,iϕ]
    A[end,iϕ] = A[end-1,iϕ]
    return
end

@parallel_indices (ir,iϕ) function bc_ϕ1!(A)
    A[ir,1  ] = A[ir,2    ]
    A[ir,end] = A[ir,end-1]
    return
end

@parallel_indices (ir,iϕ) function bc_ϕ2!(A)
    A[ir,1  ] = A[ir,end-2]
    A[ir,end] = A[ir,3    ]
    return
end

@parallel_indices (ir,iϕ) function bc_ϕ3!(A)
    A[ir,1  ] = A[ir,end-1]
    A[ir,end] = A[ir,2    ]
    return
end



@views function runme()
    # dimensionally independent
    lr,lϕ     = 1.0,2π
    dc0       = 1.0
    # scales
    tsc       = lr^2/dc0
    # nondimensional
    χ         = 2.6
    # dimensionally dependent
    ttot      = 1*tsc
    dt        = 1e-6*tsc
    r0        = 0.5*lr
    # numerics
    nr,nϕ     = 201,601
    εtol      = 1e-6
    max_iters = 20*nr
    ncheck    = ceil(Int, 0.5*nr)
    nvis      = 100
    CFL_chem  = 0.04
    # preprocessing
    dr,dϕ     = lr/nr,lϕ/nϕ
    rv,ϕv     = LinRange(r0,r0+lr,nr+1),LinRange(-dϕ,lϕ+dϕ,nϕ+1)
    rc,ϕc     = 0.5*(rv[1:end-1]+rv[2:end]),0.5*(ϕv[1:end-1]+ϕv[2:end])
    vpdτ_chem = dr*CFL_chem
    γ         = dr
    γ2        = γ^2
    c1        = π^4*(γ/lr)^2 + π^2
    c2        = c1 + lr^2/dc0/dt
    Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
    ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*lr)
    θ_dτ_chem = lr/(Re_chem*vpdτ_chem)
    # init
    C         = @zeros(nr  ,nϕ  )
    C_o       = @zeros(nr  ,nϕ  )
    qCr       = @zeros(nr+1,nϕ  )
    qCϕ       = @zeros(nr  ,nϕ+1)
    μ         = @zeros(nr  ,nϕ  )
    rC        = @zeros(nr  ,nϕ  )
    # @parallel init_C!(C,dr,dϕ,r0,lr)
    C .= 0.8.*CUDA.rand(nr,nϕ) .+ 0.1
    # action
    tcur = 0.0; it = 1
    while tcur < ttot
        @printf(" - it #%d\n", it)
        C_o .= C
        iter = 1; max_err = 2εtol; err_evo = Float64[]; iter_evo = Float64[]
        while max_err > εtol && iter < max_iters
            @parallel update_potentials!(μ,C,χ,γ2,r0,dr,dϕ)
            @parallel update_fluxes!(qCr,qCϕ,μ,C,dc0,θ_dτ_chem,r0,dr,dϕ)
            @parallel bc_ϕ2!(qCϕ); @parallel bc_ϕ3!(qCr)
            @parallel update_concentrations!(C,C_o,qCr,qCϕ,dt,ρ_dτ_chem,r0,dr,dϕ)
            @parallel bc_ϕ3!(C)
            if iter % ncheck == 0
                @parallel compute_residual!(rC,C,C_o,qCr,qCϕ,dt,r0,dr,dϕ)
                max_err = maximum(rC[:,2:end-1])*tsc
                @printf(" -- iter #%d, err (C) = %g\n", iter, max_err)
                isfinite(max_err) || error("Simulation failed")
            end
            iter += 1
        end
        if it % nvis == 0
            display(heatmap(ϕc,rc,Array(C);proj=:polar,c=:turbo))
        end
        tcur += dt; it += 1
    end
    return
end