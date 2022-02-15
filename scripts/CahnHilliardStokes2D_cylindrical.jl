using ParallelStencil
using Printf,Plots,LinearAlgebra,Statistics,MAT,Random

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
macro d_ri(A)  esc(:( $A[ir+1,iϕ+1] - $A[ir,iϕ+1] )) end
macro d_ϕi(A)  esc(:( $A[ir+1,iϕ+1] - $A[ir+1,iϕ] )) end
macro av_ra(A) esc(:( 0.5*($A[ir+1,iϕ] + $A[ir,iϕ]) )) end
macro av_ϕa(A) esc(:( 0.5*($A[ir,iϕ+1] + $A[ir,iϕ]) )) end
macro av(A) esc(:( 0.25*($A[ir,iϕ+1] + $A[ir,iϕ] + $A[ir+1,iϕ+1] + $A[ir+1,iϕ]) )) end

macro av_ri(A) esc(:( 0.5*($A[ir+1,iϕ+1] + $A[ir,iϕ+1]) )) end
macro av_ϕi(A) esc(:( 0.5*($A[ir+1,iϕ+1] + $A[ir+1,iϕ]) )) end

@parallel_indices (ir,iϕ) function init_C!(C,dr,dϕ,r0,lr)
    if checkbounds(Bool,C,ir,iϕ)
        r = r0 + (ir-1)*dr + 0.5dr
        C[ir,iϕ] = (r < r0 + 0.5lr) ? 0.999 : 0.001
    end
    return
end

@parallel_indices (ir,iϕ) function init_V!(Vr,Vϕ,dr,dϕ,r0,ϕ0,εbg)
    rc = r0 + (ir-1)*dr + 0.5*dr
    rv = rc - 0.0*dr
    ϕc = ϕ0 + (iϕ-1)*dϕ + 0.5*dϕ
    ϕv = ϕc - 0.0*dϕ

    xc = rc*cos(ϕc)
    yc = rc*sin(ϕc)
    xv = rv*cos(ϕv)
    yv = rv*sin(ϕv)

    vrx = -εbg*xc
    vry =  εbg*yv
    vϕx = -εbg*xv 
    vϕy =  εbg*yc

    if checkbounds(Bool,Vr,ir,iϕ)
        Vr[ir,iϕ] = (ir>1) *(  vrx*cos(ϕc) + vry*sin(ϕc) )
    end
    if  checkbounds(Bool,Vϕ,ir,iϕ)
        Vϕ[ir,iϕ] = (ir>1) *( -vϕx*sin(ϕc) + vϕy*cos(ϕc) )
    end

    # if checkbounds(Bool,Vr,ir,iϕ)
    #     Vr[ir,iϕ] = (  vrx*cos(ϕc) + vry*sin(ϕc) )
    # end
    # if  checkbounds(Bool,Vϕ,ir,iϕ)
    #     Vϕ[ir,iϕ] = ( -vϕx*sin(ϕc) + vϕy*cos(ϕc) )
    # end

    return
end

@parallel_indices (ir,iϕ) function compute_V_cart!(Vx,Vy,Vr,Vϕ,dr,dϕ,r0,ϕ0)
    rc = r0 + (ir-1)*dr + 0.5*dr
    rv = rc + 0.5*dr
    ϕc = ϕ0 + (iϕ-1)*dϕ + 0.5*dϕ
    ϕv = ϕc + 0.5*dϕ

    if checkbounds(Bool,Vx,ir,iϕ)
        Vx[ir,iϕ] = Vr[ir,iϕ]*cos(ϕc) - Vϕ[ir,iϕ]*sin(ϕc)
    end
    if  checkbounds(Bool,Vy,ir,iϕ)
        Vy[ir,iϕ] = Vr[ir,iϕ]*sin(ϕc) + Vϕ[ir,iϕ]*cos(ϕc)
    end

    return
end

macro Gdτ_mech()     esc(:( vpdτ_mech*Re_mech*@all(ηsτ)/lr/(r_mech+2.0) )) end
macro Gdτ_mech_av()  esc(:( vpdτ_mech*Re_mech*@av(ηsτ)/lr/(r_mech+2.0)  )) end
macro ρ_dτ_mech_r()  esc(:( Re_mech*@av_ri(ηsτ)/vpdτ_mech/lr            )) end
macro ρ_dτ_mech_ϕ()  esc(:( Re_mech*@av_ϕi(ηsτ)/vpdτ_mech/lr            )) end
macro ηs_pτ()        esc(:( 1.0/(1.0/@Gdτ_mech() + 1.0/@all(ηs))        )) end
macro ηs_pτ_av()     esc(:( 1.0/(1.0/@Gdτ_mech_av() + 1.0/@av(ηs))      )) end


@parallel_indices (ir,iϕ) function update_potentials!(μ,C,Vr,Vϕ,Pr,τrr,τϕϕ,τrϕ,ηs,ηsτ,χ,γ2,Re_mech,r_mech,vpdτ_mech,r0,lr,dr,dϕ)
    # chemical potential
    if ir <= size(μ,1) && iϕ <= size(μ,2)
        rc      = r0+(ir-1)*dr+0.5*dr
        rvn     = rc+0.5*dr
        rvs     = rc-0.5*dr
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
    # stresses
    if ir <= size(Pr,1) && iϕ <= size(Pr,2)
        rc        = r0+(ir-1)*dr+0.5*dr
        rvn       = rc+0.5*dr
        rvs       = rc-0.5*dr
        # ∇V      = ( (rvn*Vr[ir+1,iϕ] - rvs*Vr[ir,iϕ])/dr + @d_ϕa(Vϕ)/dϕ )/rc
        # @all(Pr)  = @all(Pr) - r_mech*@Gdτ_mech()*∇V
        # @all(τrr) = 2.0*@ηs_pτ()*( @d_ra(Vr)/dr - ∇V/3.0 + 0.5*@all(τrr)/@Gdτ_mech())
        # @all(τϕϕ) = 2.0*@ηs_pτ()*((@d_ϕa(Vϕ)/dϕ + @av_ra(Vr))/rc - ∇V/3.0 + 0.5*@all(τϕϕ)/@Gdτ_mech())
        Err       = @d_ra(Vr)/dr
        Eϕϕ       = @d_ϕa(Vϕ)/dϕ/rc + @av_ra(Vr)/rc
        ∇V        = Err + Eϕϕ
        @all(Pr)  = @all(Pr) - r_mech*@Gdτ_mech()*∇V
        @all(τrr) = 2.0*@ηs_pτ()*( Err - ∇V/3.0 + 0.5*@all(τrr)/@Gdτ_mech())
        @all(τϕϕ) = 2.0*@ηs_pτ()*( Eϕϕ - ∇V/3.0 + 0.5*@all(τϕϕ)/@Gdτ_mech())
    end
    if ir <= size(τrϕ,1) && iϕ <= size(τrϕ,2)
        rv = r0+ir*dr
        @all(τrϕ) = @ηs_pτ_av()*(@d_ϕi(Vr)/dϕ/rv + @d_ri(Vϕ)/dr - @av_ri(Vϕ)/rv + @all(τrϕ)/@Gdτ_mech_av())
    end
    return
end


@parallel_indices (ir,iϕ) function update_fluxes!(qCr,qCϕ,μ,C,Vr,Vϕ,Pr,τrr,τϕϕ,τrϕ,ηsτ,dc0,θ_dτ_chem,Re_mech,vpdτ_mech,r0,lr,dr,dϕ)
    # concentration flux
    if ir <= size(μ,1)-1 && iϕ <= size(μ,2)
        @inn_r(qCr) = (@inn_r(qCr)*θ_dτ_chem - dc0*@av_ra(C)*(1.0-@av_ra(C))*@d_ra(μ)/dr)/(θ_dτ_chem + 1.0)
    end
    if ir <= size(μ,1) && iϕ <= size(μ,2)-1
        rc = r0+(ir-1)*dr+0.5*dr
        @inn_ϕ(qCϕ) = (@inn_ϕ(qCϕ)*θ_dτ_chem - dc0*@av_ϕa(C)*(1.0-@av_ϕa(C))/rc*@d_ϕa(μ)/dϕ)/(θ_dτ_chem + 1.0)
    end
    # velocity
    if ir <= size(Vr,1)-2 && iϕ <= size(Vr,2)-2
        rv       = r0+ir*dr
        rcn      = rv+0.5*dr
        rcs      = rv-0.5*dr
        # @inn(Vr) = @inn(Vr) + (-@d_ri(Pr)/dr + (rcn*τrr[ir+1,iϕ+1] - rcs*τrr[ir,iϕ+1])/dr/rv + @d_ϕa(τrϕ)/dϕ/rv - @av_ri(τϕϕ)/rv)/@ρ_dτ_mech_r()
        @inn(Vr) = @inn(Vr) + (-@d_ri(Pr)/dr + @d_ri(τrr)/dr + @d_ϕa(τrϕ)/dϕ/rv + (@av_ri(τrr) - @av_ri(τϕϕ))/rv)/@ρ_dτ_mech_r()
    end
    if ir <= size(Vϕ,1)-2 && iϕ <= size(Vϕ,2)-2
        rc      = r0+ir*dr+0.5*dr
        rvs     = rc-0.5*dr 
        rvn     = rc+0.5*dr
        # @inn(Vϕ) = @inn(Vϕ) + (-@d_ϕi(Pr)/dϕ/rc + @d_ϕi(τϕϕ)/dϕ/rc + (rvn^2*τrϕ[ir+1,iϕ]-rvs^2*τrϕ[ir,iϕ])/dr/rc^2)/@ρ_dτ_mech_ϕ()
        @inn(Vϕ) = @inn(Vϕ) + (-@d_ϕi(Pr)/dϕ/rc + @d_ϕi(τϕϕ)/dϕ/rc + @d_ra(τrϕ)/dr + 2.0*@av_ra(τrϕ)/rc)/@ρ_dτ_mech_ϕ()
    end
    return
end


@parallel_indices (ir,iϕ) function update_concentrations!(C,C_o,qCr,qCϕ,ηs,dt,ρ_dτ_chem,ηs0,npow,r0,dr,dϕ)
    if checkbounds(Bool,C,ir,iϕ)
        rc       = r0+(ir-1)*dr+0.5*dr
        rvn      = rc+0.5*dr
        rvs      = rc-0.5*dr
        @all(C)  = (@all(C)*ρ_dτ_chem + @all(C_o)/dt - (rvn*qCr[ir+1,iϕ]-rvs*qCr[ir,iϕ])/dr/rc - @d_ϕa(qCϕ)/dϕ/rc)/(ρ_dτ_chem + 1.0/dt)
        ηs_rel   = 1e-2
        @all(ηs) = @all(ηs)*(1.0-ηs_rel) + ηs0*10.0^(npow*@all(C))*ηs_rel
        # @all(ηs) = ηs0*10.0^(npow*@all(C))
    end
    return
end


@parallel_indices (ir,iϕ) function advect_C!(dC_dt,C,Vr,Vϕ,r0,dr,dϕ)
    if ir<=size(dC_dt, 1) && iϕ<=size(dC_dt, 2)
        rc = r0 + ir*dr+0.5*dr
        dC_dt[ir,iϕ] = - max(Vr[ir+1,iϕ+1],0.0)*(C[ir+1,iϕ+1]-C[ir  ,iϕ+1])/dr -
                         min(Vr[ir+2,iϕ+1],0.0)*(C[ir+2,iϕ+1]-C[ir+1,iϕ+1])/dr -
                         max(Vϕ[ir+1,iϕ+1],0.0)*(C[ir+1,iϕ+1]-C[ir+1,iϕ  ])/dϕ/rc -
                         min(Vϕ[ir+1,iϕ+2],0.0)*(C[ir+1,iϕ+2]-C[ir+1,iϕ+1])/dϕ/rc
    end
    return
end


@parallel_indices (ir,iϕ) function compute_true_τ!(τrr2,τϕϕ2,τrϕ2,∇V,Vr,Vϕ,ηs,r0,dr,dϕ)
    if ir <= size(τrr2,1) && iϕ <= size(τrr2,2)
        rc        = r0+(ir-1)*dr+0.5*dr
        rvn       = rc+0.5*dr
        rvs       = rc-0.5*dr
        # ∇V[ir,iϕ]  = ( (rvn*Vr[ir+1,iϕ] - rvs*Vr[ir,iϕ])/dr + @d_ϕa(Vϕ)/dϕ )/rc
        # @all(τrr2) = 2.0*@all(ηs)*( @d_ra(Vr)/dr - ∇V[ir,iϕ]/3.0)
        # @all(τϕϕ2) = 2.0*@all(ηs)*((@d_ϕa(Vϕ)/dϕ + @av_ra(Vr))/rc - ∇V[ir,iϕ]/3.0)
        Err       = @d_ra(Vr)/dr
        Eϕϕ       = @d_ϕa(Vϕ)/dϕ/rc + @av_ra(Vr)/rc
        ∇V[ir,iϕ] = Err + Eϕϕ
        @all(τrr2) = 2.0*@all(ηs)*( Err - ∇V[ir,iϕ]/3.0 )
        @all(τϕϕ2) = 2.0*@all(ηs)*( Eϕϕ - ∇V[ir,iϕ]/3.0 )
    end
    if ir <= size(τrϕ2,1) && iϕ <= size(τrϕ2,2)
        rv = r0+ir*dr
        @all(τrϕ2) = @av(ηs)*(@d_ϕi(Vr)/dϕ/rv + @d_ri(Vϕ)/dr - @av_ri(Vϕ)/rv)
    end
    return
end


@parallel_indices (ir,iϕ) function compute_residuals!(rC,C,C_o,qCr,qCϕ,rVr,rVϕ,Pr,τrr2,τϕϕ2,τrϕ2,dt,r0,dr,dϕ)
    # chemistry
    if ir <= size(C,1) && iϕ <= size(C,2)
        rc       = r0 + (ir-1.0)*dr + 0.5*dr
        rvn      = rc + 0.5*dr
        rvs      = rc - 0.5*dr
        @all(rC) = abs( (@all(C) - @all(C_o))/dt + (rvn*qCr[ir+1,iϕ]-rvs*qCr[ir,iϕ])/dr/rc + @d_ϕa(qCϕ)/dϕ/rc )
    end
    # mechanics
    if ir <= size(rVr,1) && iϕ <= size(rVr,2)
        rv        = r0 + ir*dr
        rcn       = rv + 0.5*dr
        rcs       = rv - 0.5*dr
        # @all(rVr) = abs( -@d_ri(Pr)/dr + (rcn*τrr2[ir+1,iϕ+1] - rcs*τrr2[ir,iϕ+1])/dr/rv + @d_ϕa(τrϕ2)/dϕ/rv - @av_ri(τϕϕ2)/rv )
        @all(rVr) = abs( -@d_ri(Pr)/dr + @d_ri(τrr2)/dr + @d_ϕa(τrϕ2)/dϕ/rv + (@av_ri(τrr2) - @av_ri(τϕϕ2))/rv )
    end
    if ir <= size(rVϕ,1) && iϕ <= size(rVϕ,2)
        rc2       = r0 + ir*dr + 0.5*dr
        rvn2      = rc2 + 0.5*dr
        rvs2      = rc2 - 0.5*dr
        # @all(rVϕ) = abs( -@d_ϕi(Pr)/dϕ/rc2 + @d_ϕi(τϕϕ2)/dϕ/rc2 + (rvn2^2*τrϕ2[ir+1,iϕ]-rvs2^2*τrϕ2[ir,iϕ])/dr/rc2^2 )
        @all(rVϕ) = abs( -@d_ϕi(Pr)/dϕ/rc2 + @d_ϕi(τϕϕ2)/dϕ/rc2 + @d_ra(τrϕ2)/dr + 2.0*@av_ra(τrϕ2)/rc2 )
    end
    return
end

# @parallel_indices (ir,iϕ) function smooth(A2,A)

#     return
# end


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
    ηs0       = 1.0
    # scales
    tsc       = lr^2/dc0
    vsc       = dc0/lr
    # nondimensional
    χ         = 2.6
    Pe        = 0.0
    npow      = 3.0
    # dimensionally dependent
    ttot      = 1*tsc
    dt        = 1e-5*tsc
    r0        = 0.5*lr
    εbg       = 10.0/tsc
    # numerics
    nr,nϕ     = 201,601
    εtol      = 1e-4
    max_iters = 200*nr
    ncheck    = ceil(Int, 10nr)
    nvis      = 10
    CFL_chem  = 0.05
    CFL_mech  = 0.6/sqrt(2)
    # preprocessing
    dr,dϕ     = lr/nr,lϕ/(nϕ-2)
    rv,ϕv     = LinRange(r0,r0+lr,nr+1),LinRange(-dϕ,lϕ+dϕ,nϕ+1)
    rc,ϕc     = 0.5*(rv[1:end-1]+rv[2:end]),0.5*(ϕv[1:end-1]+ϕv[2:end])
    vpdτ_chem = dr*CFL_chem
    vpdτ_mech = dr*CFL_mech
    Re_mech   = 5π
    r_mech    = 1.0
    γ         = 2dr
    γ2        = γ^2
    c1        = π^4*(γ/lr)^2 + π^2
    c2        = c1 + lr^2/dc0/dt
    Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
    ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*lr)
    θ_dτ_chem = lr/(Re_chem*vpdτ_chem)
    # init
    C         = @zeros(nr  ,nϕ  )
    C_o       = @zeros(nr  ,nϕ  )
    dC_dt     = @zeros(nr-2,nϕ-2)
    qCr       = @zeros(nr+1,nϕ  )
    qCϕ       = @zeros(nr  ,nϕ+1)
    μ         = @zeros(nr  ,nϕ  )
    rC        = @zeros(nr  ,nϕ  )
    Pr        = @zeros(nr  ,nϕ  )
    Pr_o      = @zeros(nr  ,nϕ  )
    τrr       = @zeros(nr  ,nϕ  )
    τϕϕ       = @zeros(nr  ,nϕ  )
    τrϕ       = @zeros(nr-1,nϕ-1)
    Vr        = @zeros(nr+1,nϕ  )
    Vϕ        = @zeros(nr  ,nϕ+1)
    Vx        = @zeros(nr  ,nϕ  )
    Vy        = @zeros(nr  ,nϕ  )
    ηs        = @zeros(nr  ,nϕ  )
    ηsτ       = @zeros(nr  ,nϕ  )
    τrr2      = @zeros(nr  ,nϕ  )
    τϕϕ2      = @zeros(nr  ,nϕ  )
    τrϕ2      = @zeros(nr-1,nϕ-1)
    ∇V        = @zeros(nr  ,nϕ  )
    rVr       = @zeros(nr-1,nϕ-2)
    rVϕ       = @zeros(nr-2,nϕ-1)
    # @parallel init_C!(C,dr,dϕ,r0,lr)
    C        .= 0.9.*CUDA.rand(nr,nϕ) .+ 0.05
    @parallel init_V!(Vr,Vϕ,dr,dϕ,r0,-dϕ,εbg)
    ηs       .= ηs0#.*10.0 .^ (npow.*C)
    ηsτ      .= ηs
    # action
    tcur = 0.0; it = 1
    while tcur < ttot
        @printf(" - it #%d\n", it)
        C_o  .= C
        Pr_o .= Pr
        iter = 1; max_err = 2εtol*ones(4); err_evo = Float64[]; iter_evo = Float64[]
        while any(max_err .> εtol) && iter < max_iters
            @parallel update_potentials!(μ,C,Vr,Vϕ,Pr,τrr,τϕϕ,τrϕ,ηs,ηsτ,χ,γ2,Re_mech,r_mech,vpdτ_mech,r0,lr,dr,dϕ)
            @parallel update_fluxes!(qCr,qCϕ,μ,C,Vr,Vϕ,Pr,τrr,τϕϕ,τrϕ,ηsτ,dc0,θ_dτ_chem,Re_mech,vpdτ_mech,r0,lr,dr,dϕ)
            @parallel bc_ϕ2!(qCϕ); @parallel bc_ϕ3!(qCr); @parallel bc_ϕ2!(Vϕ); @parallel bc_ϕ3!(Vr)
            @parallel update_concentrations!(C,C_o,qCr,qCϕ,ηs,dt,ρ_dτ_chem,ηs0,npow,r0,dr,dϕ)
            @parallel bc_ϕ3!(C); @parallel bc_ϕ3!(ηs)
            ηsτ .= ηs
            if iter % ncheck == 0
                Pr .-= mean(Pr)
                @parallel compute_true_τ!(τrr2,τϕϕ2,τrϕ2,∇V,Vr,Vϕ,ηs,r0,dr,dϕ)
                @parallel compute_residuals!(rC,C,C_o,qCr,qCϕ,rVr,rVϕ,Pr,τrr2,τϕϕ2,τrϕ2,dt,r0,dr,dϕ)
                max_err[1] = maximum(rC[:,2:end-1])*tsc
                max_err[2] = maximum(rVr[:,2:end-1])*tsc/vsc
                max_err[3] = maximum(rVϕ[:,2:end-1])*tsc/vsc
                max_err[4] = maximum(abs.(∇V[2:end-1,2:end-1]))*tsc
                @printf(" -- iter/nr = %g, err (C) = %g, err (Vr) = %g, err (Vϕ) = %g, err (∇V) = %g\n", iter/nr, max_err...)
                all(isfinite.(max_err)) || error("Simulation failed")
            end
            iter += 1
        end
        @parallel advect_C!(dC_dt,C,Vr,Vϕ,r0,dr,dϕ)
        C[2:end-1,2:end-1] .+= dt*dC_dt
        if it % nvis == 0
            @parallel compute_V_cart!(Vx,Vy,Vr,Vϕ,dr,dϕ,r0,-dϕ)
            p1 = heatmap(ϕc,rc,Array(Vx);proj=:polar,c=:turbo,title="Vx")
            p2 = heatmap(ϕc,rc,Array(Vy);proj=:polar,c=:turbo,title="Vy")
            p3 = heatmap(ϕc,rc,Array(Pr);proj=:polar,c=:turbo,title="P")
            p4 = heatmap(ϕc,rc,Array(C);proj=:polar,c=:turbo,title="C")
            display(plot(p1,p2,p3,p4;layout=(2,2),size=(1e3,600),dpi=100))
        end
        tcur += dt; it += 1

        if it > 2
            dt        = 1e-4*tsc
            c1        = π^4*(γ/lr)^2 + π^2
            c2        = c1 + lr^2/dc0/dt
            Re_chem   = sqrt(c1 + c2 + 2*sqrt(c1*c2))
            ρ_dτ_chem = Re_chem*dc0/(vpdτ_chem*lr)
            θ_dτ_chem = lr/(Re_chem*vpdτ_chem)
        end
    end
    return
end