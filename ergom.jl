using LinearAlgebra
using Printf

const secs_pr_day = 86400

# Helper functions
function osat_weiss(t, s)
    aa1, aa2, a3, a4 = -173.4292, 249.6339, 143.3483, -21.8492
    b1, b2, b3 = -0.033096, 0.014259, -0.001700
    kelvin, mol_per_liter = 273.16, 44.661

    tk = (t + kelvin) * 0.01
    return exp(aa1 + aa2 / tk + a3 * log(tk) + a4 * tk + s * (b1 + (b2 + b3 * tk) * tk)) * mol_per_liter
end

function th(x, w, min, max)
    if w > 1.0e-10
        return min + (max - min) * 0.5 * (1.0 + tanh(x / w))
    elseif x > 0.0
        return 1.0
    else
        return 0.0
    end
end

function yy(a, x)
    return x^2 / (a^2 + x^2)
end

function fpz(iv, g, t, topt, psum)
    return g * (1.0 + t^2 / topt^2 * exp(1.0 - 2.0 * t / topt)) * (1.0 - exp(-(iv^2) * psum^2))
end

# ERGOM Model
mutable struct ERGOM
    # Parameters
    sfl_po::Float64
    sfl_ni::Float64
    sfl_am::Float64
    fluff::Bool
    p10::Float64
    p20::Float64
    p30::Float64
    zo0::Float64
    kc::Float64
    i_min::Float64
    r1max::Float64
    r2max::Float64
    r3max::Float64
    alpha1::Float64
    alpha2::Float64
    alpha3::Float64
    lpa::Float64
    lpd::Float64
    tf::Float64
    tbg::Float64
    beta_bg::Float64
    g1max::Float64
    g2max::Float64
    g3max::Float64
    lza::Float64
    lzd::Float64
    iv::Float64
    topt::Float64
    lan::Float64
    oan::Float64
    beta_an::Float64
    lda::Float64
    tda::Float64
    beta_da::Float64
    pvel::Float64
    sr::Float64
    s1::Float64
    s2::Float64
    s3::Float64
    s4::Float64
    a0::Float64
    a1::Float64
    a2::Float64
    lds::Float64
    lsd::Float64
    tau_crit::Float64
    lsa::Float64
    bsa::Float64
    ph1::Float64
    ph2::Float64
    w_p1::Float64
    w_p2::Float64
    w_p3::Float64
    w_de::Float64

    # State variables
    p1::Float64
    p2::Float64
    p3::Float64
    zo::Float64
    de::Float64
    o2::Float64
    am::Float64
    ni::Float64
    po::Float64
    fl::Float64

    # Dependencies
    I_0::Float64
    par::Float64
    temp::Float64
    salt::Float64
    wind::Float64
    taub::Float64

    function ERGOM()
        return new(
            0.0015 / secs_pr_day, 0.09 / secs_pr_day, 0.07 / secs_pr_day, false,
            0.0225, 0.0225, 0.0225, 0.0225, 0.03, 25.0, 2.0 / secs_pr_day, 0.7 / secs_pr_day, 0.5 / secs_pr_day,
            1.35, 0.675, 0.5, 0.01 / secs_pr_day, 0.02 / secs_pr_day, 10.0, 14.0, 1.0, 0.5 / secs_pr_day,
            0.5 / secs_pr_day, 0.25 / secs_pr_day, 0.0666666666 / secs_pr_day, 0.1333333333 / secs_pr_day,
            0.24444444, 20.0, 0.1 / secs_pr_day, 0.01, 0.11, 0.003 / secs_pr_day, 13.0, 20.0, 5.0 / secs_pr_day,
            0.0625, 5.3, 6.625, 8.625, 2.0, 31.25, 14.603, 0.4025, 3.5 / secs_pr_day, 25.0 / secs_pr_day,
            0.07, 0.001 / secs_pr_day, 0.15, 0.15, 0.1, -1.0 / secs_pr_day, -5.0 / secs_pr_day, -5.0 / secs_pr_day,
            -3.0 / secs_pr_day, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 0.0, 50.0, 10.0, 15.0, 35.0, 5.0, 0.0
        )
    end
end

function process_interior(mod::ERGOM)
    wo, wn = 30.0, 0.1

    # Light acclimation formulation based on surface light intensity
    iopt = max(0.25 * mod.I_0, mod.i_min)
    ppi = mod.par / iopt * exp(1.0 - mod.par / iopt)

    thopnp = th(mod.o2, wo, 0.0, 1.0) * yy(wn, mod.ni)
    thomnp = th(-mod.o2, wo, 0.0, 1.0) * yy(wn, mod.ni)
    thomnm = th(-mod.o2, wo, 0.0, 1.0) * (1.0 - yy(wn, mod.ni))
    thsum = thopnp + thomnp + thomnm
    thopnp /= thsum
    thomnp /= thsum
    thomnm /= thsum

    psum = mod.p1 + mod.p2 + mod.p3 + mod.p10 + mod.p20 + mod.p30

    llda = mod.lda * (1.0 + mod.beta_da * yy(mod.tda, mod.temp))
    llan = th(mod.o2, 0.0, 0.0, 1.0) * mod.o2 / (mod.oan + mod.o2) * mod.lan * exp(mod.beta_an * mod.temp)

    lp = mod.lpa + mod.lpd

    r1 = mod.r1max * min(yy(mod.alpha1, mod.am + mod.ni), yy(mod.sr * mod.alpha1, mod.po), ppi) * (mod.p1 + mod.p10)
    r2 = mod.r2max * (1.0 + yy(mod.tf, mod.temp)) * min(yy(mod.alpha2, mod.am + mod.ni), yy(mod.sr * mod.alpha2, mod.po), ppi) * (mod.p2 + mod.p20)
    r3 = mod.r3max * 1.0 / (1.0 + exp(mod.beta_bg * (mod.tbg - mod.temp))) * min(yy(mod.sr * mod.alpha3, mod.po), ppi) * (mod.p3 + mod.p30)

    mod.p1 += r1 - fpz(mod.iv, mod.g1max, mod.temp, mod.topt, psum) * mod.p1 / psum * (mod.zo + mod.zo0) - lp * mod.p1
    mod.p2 += r2 - fpz(mod.iv, mod.g2max, mod.temp, mod.topt, psum) * mod.p2 / psum * (mod.zo + mod.zo0) - lp * mod.p2
    mod.p3 += r3 - fpz(mod.iv, mod.g3max, mod.temp, mod.topt, psum) * mod.p3 / psum * (mod.zo + mod.zo0) - lp * mod.p3
    mod.zo += (fpz(mod.iv, mod.g1max, mod.temp, mod.topt, psum) * mod.p1 + fpz(mod.iv, mod.g2max, mod.temp, mod.topt, psum) * mod.p2 + fpz(mod.iv, mod.g3max, mod.temp, mod.topt, psum) * mod.p3) * (mod.zo + mod.zo0) / psum - mod.lza * mod.zo * (mod.zo + mod.zo0) - mod.lzd * mod.zo * (mod.zo + mod.zo0)
    mod.de += mod.lpd * (mod.p1 + mod.p2 + mod.p3) + mod.lzd * mod.zo * (mod.zo + mod.zo0) - llda * mod.de
    mod.am += llda * mod.de - llan * mod.am - mod.am / (mod.am + mod.ni) * (r1 + r2) + mod.lpa * (mod.p1 + mod.p2 + mod.p3) + mod.lza * mod.zo * (mod.zo + mod.zo0)
    mod.ni += llan * mod.am - mod.ni / (mod.am + mod.ni) * (r1 + r2) - mod.s1 * llda * mod.de * thomnp
    mod.po += mod.sr * (-r1 - r2 - r3 + llda * mod.de + mod.lpa * (mod.p1 + mod.p2 + mod.p3) + mod.lza * mod.zo * (mod.zo + mod.zo0))
    mod.o2 += mod.s2 * (mod.am / (mod.am + mod.ni) * (r1 + r2) + r3) - mod.s4 * (llan * mod.am) + mod.s3 * (mod.ni / (mod.am + mod.ni) * (r1 + r2)) - mod.s2 * (thopnp + thomnm) * llda * mod.de - mod.s2 * (mod.lpa * (mod.p1 + mod.p2 + mod.p3) + mod.lza * mod.zo * (mod.zo + mod.zo0))
end

function process_bottom(mod::ERGOM)
    wo, wn, dot2 = 30.0, 0.1, 0.2

    thopnp = th(mod.o2, wo, 0.0, 1.0) * yy(wn, mod.ni)
    thomnp = th(-mod.o2, wo, 0.0, 1.0) * yy(wn, mod.ni)
    thomnm = th(-mod.o2, wo, 0.0, 1.0) * (1.0 - yy(wn, mod.ni))
    thsum = thopnp + thomnp + thomnm
    thopnp /= thsum
    thomnp /= thsum
    thomnm /= thsum

    llsa = mod.lsa * exp(mod.bsa * mod.temp) * th(mod.o2, wo, dot2, 1.0)

    # Sedimentation dependent on bottom stress
    llds = mod.tau_crit > mod.taub ? mod.lds * (mod.tau_crit - mod.taub) / mod.tau_crit : 0.0

    # Resuspension dependent on bottom stress
    llsd = mod.tau_crit < mod.taub ? mod.lsd * (mod.taub - mod.tau_crit) / mod.tau_crit : 0.0

    mod.fl += llds * mod.de - llsd * mod.fl - llsa * mod.fl - th(mod.o2, wo, 0.0, 1.0) * llsa * mod.fl
    mod.de -= llds * mod.de + llsd * mod.fl
    mod.am += llsa * mod.fl
    mod.ni -= mod.s1 * thomnp * llsa * mod.fl
    mod.po += mod.sr * (1.0 - mod.ph1 * th(mod.o2, wo, 0.0, 1.0) * yy(mod.ph2, mod.o2)) * llsa * mod.fl
    mod.o2 -= (mod.s4 + mod.s2 * (thopnp + thomnm)) * llsa * mod.fl
end

function process_surface(mod::ERGOM)
    newflux = 1
    if newflux == 1
        sc = 1450.0 + (1.1 * mod.temp - 71.0) * mod.temp
        if mod.wind > 13.0
            p_vel = 5.9 * (5.9 * mod.wind - 49.3) / sqrt(sc)
        elseif mod.wind < 3.6
            p_vel = 1.003 * mod.wind / (sc)^(0.66)
        else
            p_vel = 5.9 * (2.85 * mod.wind - 9.65) / sqrt(sc)
        end
        p_vel = max(p_vel, 0.05) / secs_pr_day
        flo2 = p_vel * (osat_weiss(mod.temp, mod.salt) - mod.o2)
    else
        flo2 = mod.pvel * (mod.a0 * (mod.a1 - mod.a2 * mod.temp) - mod.o2)
    end
    mod.o2 += flo2

    mod.ni += mod.sfl_ni
    mod.am += mod.sfl_am
    mod.po += mod.sfl_po
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    mod = ERGOM()
    process_interior(mod)
    process_bottom(mod)
    process_surface(mod)

    println("Simulation complete.")
end
