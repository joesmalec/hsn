//! `exp`, `erf` and GELU for the CPU oracle, in pure Rust.
//!
//! The oracle must produce the same bytes on every platform (the test vectors are regenerated
//! and compared byte for byte on Linux x86_64 and macOS arm64). `f64::exp` calls the platform
//! libm, whose last-ulp results differ between glibc and Apple's libm, and the standard library
//! has no `erf`. These are ports of fdlibm 5.3's `__ieee754_exp` (`e_exp.c`) and `erf`
//! (`s_erf.c`), accurate to under one ulp, built from IEEE `+ − × ÷` and bit operations only
//! (Rust never contracts them into fused multiply-adds), so their results are identical
//! everywhere.
//!
//! The fdlibm sources carry this notice, which their use here preserves:
//!
//! > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! >
//! > Developed at SunSoft, a Sun Microsystems, Inc. business.
//! > Permission to use, copy, modify, and distribute this
//! > software is freely granted, provided that this notice
//! > is preserved.

// The constants are fdlibm's decimal literals, verbatim (21 significant digits), so that they can
// be checked against the source; they parse to the same doubles as fdlibm's hex comments.
#![allow(clippy::excessive_precision)]

/// The high 32 bits of `x` (fdlibm's `__HI`).
fn hi(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

/// The low 32 bits of `x` (fdlibm's `__LO`).
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}

/// `x` with its high word increased by `k << 20`, i.e. multiplied by `2^k` for normal results.
fn add_to_exponent(x: f64, k: i32) -> f64 {
    f64::from_bits(x.to_bits().wrapping_add(((k as i64) << 52) as u64))
}

const LN2_HI: [f64; 2] = [
    6.931_471_803_691_238_164_90e-01,
    -6.931_471_803_691_238_164_90e-01,
];
const LN2_LO: [f64; 2] = [
    1.908_214_929_270_587_700_02e-10,
    -1.908_214_929_270_587_700_02e-10,
];
const HALF: [f64; 2] = [0.5, -0.5];
/// fdlibm's `invln2` (1.44269504088896338700e+00), the same double as `LOG2_E`.
const INV_LN2: f64 = std::f64::consts::LOG2_E;
const O_THRESHOLD: f64 = 7.097_827_128_933_839_730_96e+02;
const U_THRESHOLD: f64 = -7.451_332_191_019_411_084_20e+02;
const TWO_M1000: f64 = 9.332_636_185_032_188_789_90e-302;
const HUGE: f64 = 1.0e300;
const P1: f64 = 1.666_666_666_666_660_190_37e-01;
const P2: f64 = -2.777_777_777_701_559_338_42e-03;
const P3: f64 = 6.613_756_321_437_934_361_17e-05;
const P4: f64 = -1.653_390_220_546_525_153_90e-06;
const P5: f64 = 4.138_136_797_057_238_460_39e-08;

/// `e^x` (fdlibm `__ieee754_exp`).
pub fn exp(x: f64) -> f64 {
    let mut hx = hi(x);
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7fff_ffff;

    // Non-finite and out-of-range arguments.
    if hx >= 0x4086_2e42 {
        if hx >= 0x7ff0_0000 {
            if ((hx & 0x000f_ffff) | lo(x)) != 0 {
                return x + x; // NaN
            }
            return if xsb == 0 { x } else { 0.0 }; // exp(±inf) = {inf, 0}
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow
        }
        if x < U_THRESHOLD {
            return TWO_M1000 * TWO_M1000; // underflow
        }
    }

    // Argument reduction: x = k·ln2 + r, |r| <= 0.5·ln2, r = hi − lo.
    let (mut r, hi_part, lo_part, k): (f64, f64, f64, i32);
    if hx > 0x3fd6_2e42 {
        if hx < 0x3ff0_a2b2 {
            hi_part = x - LN2_HI[xsb];
            lo_part = LN2_LO[xsb];
            k = 1 - (xsb as i32) - (xsb as i32);
        } else {
            k = (INV_LN2 * x + HALF[xsb]) as i32;
            let t = k as f64;
            hi_part = x - t * LN2_HI[0]; // t·ln2HI is exact here
            lo_part = t * LN2_LO[0];
        }
        r = hi_part - lo_part;
    } else if hx < 0x3e30_0000 {
        // |x| < 2^-28
        return 1.0 + x;
    } else {
        k = 0;
        hi_part = 0.0;
        lo_part = 0.0;
        r = x;
    }

    // r is in the primary range.
    let t = r * r;
    let c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((r * c) / (c - 2.0) - r);
    }
    let y = 1.0 - ((lo_part - (r * c) / (2.0 - c)) - hi_part);
    r = y;
    if k >= -1021 {
        add_to_exponent(r, k)
    } else {
        add_to_exponent(r, k + 1000) * TWO_M1000
    }
}

const ERX: f64 = 8.450_629_115_104_675_292_97e-01;
const EFX: f64 = 1.283_791_670_955_125_863_16e-01;
const EFX8: f64 = 1.027_033_336_764_100_690_53e+00;
const PP: [f64; 5] = [
    1.283_791_670_955_125_585_61e-01,
    -3.250_421_072_470_014_993_70e-01,
    -2.848_174_957_559_851_047_66e-02,
    -5.770_270_296_489_441_591_57e-03,
    -2.376_301_665_665_016_260_84e-05,
];
const QQ: [f64; 5] = [
    3.979_172_239_591_553_528_19e-01,
    6.502_224_998_876_729_444_85e-02,
    5.081_306_281_875_765_627_76e-03,
    1.324_947_380_043_216_445_26e-04,
    -3.960_228_278_775_368_123_20e-06,
];
const PA: [f64; 7] = [
    -2.362_118_560_752_659_440_77e-03,
    4.148_561_186_837_483_316_66e-01,
    -3.722_078_760_357_013_238_47e-01,
    3.183_466_199_011_617_536_74e-01,
    -1.108_946_942_823_966_774_76e-01,
    3.547_830_432_561_823_593_71e-02,
    -2.166_375_594_868_790_843_00e-03,
];
const QA: [f64; 6] = [
    1.064_208_804_008_442_282_86e-01,
    5.403_979_177_021_710_489_37e-01,
    7.182_865_441_419_626_628_68e-02,
    1.261_712_198_087_616_421_12e-01,
    1.363_708_391_202_905_073_62e-02,
    1.198_449_984_679_910_741_70e-02,
];
const RA: [f64; 8] = [
    -9.864_944_034_847_148_227_05e-03,
    -6.938_585_727_071_817_643_72e-01,
    -1.055_862_622_532_329_098_14e+01,
    -6.237_533_245_032_600_603_96e+01,
    -1.623_966_694_625_734_703_55e+02,
    -1.846_050_929_067_110_359_94e+02,
    -8.128_743_550_630_659_342_46e+01,
    -9.814_329_344_169_145_485_92e+00,
];
const SA: [f64; 8] = [
    1.965_127_166_743_925_712_92e+01,
    1.376_577_541_435_190_426_00e+02,
    4.345_658_774_752_292_288_21e+02,
    6.453_872_717_332_678_803_36e+02,
    4.290_081_400_275_678_333_86e+02,
    1.086_350_055_417_794_351_34e+02,
    6.570_249_770_319_281_701_35e+00,
    -6.042_441_521_485_809_874_38e-02,
];
const RB: [f64; 7] = [
    -9.864_942_924_700_099_285_97e-03,
    -7.992_832_376_805_230_065_74e-01,
    -1.775_795_491_775_475_198_89e+01,
    -1.606_363_848_558_219_160_62e+02,
    -6.375_664_433_683_896_277_22e+02,
    -1.025_095_131_611_077_249_54e+03,
    -4.835_191_916_086_513_970_19e+02,
];
const SB: [f64; 7] = [
    3.033_806_074_348_245_829_24e+01,
    3.257_925_129_965_739_188_26e+02,
    1.536_729_586_084_436_959_94e+03,
    3.199_858_219_508_595_539_08e+03,
    2.553_050_406_433_164_425_83e+03,
    4.745_285_412_069_553_672_15e+02,
    -2.244_095_244_658_581_833_62e+01,
];
const TINY: f64 = 1e-300;

/// The error function (fdlibm `erf`).
pub fn erf(x: f64) -> f64 {
    let hx = hi(x) as i32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // erf(nan) = nan, erf(±inf) = ±1
        let i = (((hx as u32) >> 31) << 1) as i32;
        return (1 - i) as f64 + 1.0 / x;
    }

    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3e30_0000 {
            // |x| < 2^-28
            if ix < 0x0080_0000 {
                return 0.125 * (8.0 * x + EFX8 * x); // avoid underflow
            }
            return x + EFX * x;
        }
        let z = x * x;
        let r = PP[0] + z * (PP[1] + z * (PP[2] + z * (PP[3] + z * PP[4])));
        let s = 1.0 + z * (QQ[0] + z * (QQ[1] + z * (QQ[2] + z * (QQ[3] + z * QQ[4]))));
        let y = r / s;
        return x + x * y;
    }
    if ix < 0x3ff4_0000 {
        // 0.84375 <= |x| < 1.25
        let s = x.abs() - 1.0;
        let p =
            PA[0] + s * (PA[1] + s * (PA[2] + s * (PA[3] + s * (PA[4] + s * (PA[5] + s * PA[6])))));
        let q =
            1.0 + s * (QA[0] + s * (QA[1] + s * (QA[2] + s * (QA[3] + s * (QA[4] + s * QA[5])))));
        return if hx >= 0 { ERX + p / q } else { -ERX - p / q };
    }
    if ix >= 0x4018_0000 {
        // |x| >= 6
        return if hx >= 0 { 1.0 - TINY } else { TINY - 1.0 };
    }
    let ax = x.abs();
    let s = 1.0 / (ax * ax);
    let (r, big_s);
    if ix < 0x4006_db6e {
        // |x| < 1/0.35
        r = RA[0]
            + s * (RA[1]
                + s * (RA[2] + s * (RA[3] + s * (RA[4] + s * (RA[5] + s * (RA[6] + s * RA[7]))))));
        big_s = 1.0
            + s * (SA[0]
                + s * (SA[1]
                    + s * (SA[2]
                        + s * (SA[3] + s * (SA[4] + s * (SA[5] + s * (SA[6] + s * SA[7])))))));
    } else {
        // |x| >= 1/0.35
        r = RB[0] + s * (RB[1] + s * (RB[2] + s * (RB[3] + s * (RB[4] + s * (RB[5] + s * RB[6])))));
        big_s = 1.0
            + s * (SB[0]
                + s * (SB[1] + s * (SB[2] + s * (SB[3] + s * (SB[4] + s * (SB[5] + s * SB[6]))))));
    }
    let z = f64::from_bits(ax.to_bits() & 0xffff_ffff_0000_0000); // low word cleared
    let rr = exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + r / big_s);
    if hx >= 0 {
        1.0 - rr / ax
    } else {
        rr / ax - 1.0
    }
}

/// `1/√2`.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1/√(2π)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_94;

/// The standard normal CDF `Φ(x) = (1 + erf(x/√2)) / 2`.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// The standard normal density `φ(x) = e^{−x²/2} / √(2π)`.
pub fn normal_pdf(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * exp(-0.5 * x * x)
}

/// The exact GELU, `x·Φ(x) = x/2 · (1 + erf(x/√2))` (`torch.nn.GELU()`).
pub fn gelu(x: f64) -> f64 {
    x * normal_cdf(x)
}

/// `gelu'(x) = Φ(x) + x·φ(x)`.
pub fn gelu_grad(x: f64) -> f64 {
    normal_cdf(x) + x * normal_pdf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in units in the last place (both finite, same sign or zero).
    fn ulps(a: f64, b: f64) -> u64 {
        let key = |x: f64| {
            let bits = x.to_bits() as i64;
            if bits < 0 { i64::MIN - bits } else { bits }
        };
        key(a).abs_diff(key(b))
    }

    #[test]
    fn test_p2_9_exp_agrees_with_libm_within_one_ulp() {
        // libm's exp is itself within one ulp of the true value; agreement to one ulp catches any
        // transcription error in the reduction or the polynomial.
        let mut worst = 0;
        for i in -20_000..=20_000 {
            let x = i as f64 * 0.003_706_5; // covers [-74, 74] and both reduction branches
            worst = worst.max(ulps(exp(x), x.exp()));
        }
        for x in [
            0.0, -0.0, 1e-30, -1e-30, 0.34, 0.35, 1.03, 1.04, 700.0, -700.0, -740.0,
        ] {
            worst = worst.max(ulps(exp(x), x.exp()));
        }
        assert!(worst <= 1, "exp differs from libm by {worst} ulp");
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0), 0.0);
    }

    #[test]
    fn test_p2_9_erf_matches_reference_values() {
        // Reference values: CPython's math.erf (glibc), printed with repr.
        let reference = [
            (0.0, 0.0),
            (1e-10, 1.1283791670955126e-10),
            (0.1, 0.1124629160182849),
            (0.25, 0.27632639016823696),
            (0.5, 0.5204998778130465),
            (0.75, 0.7111556336535151),
            (0.84375, 0.7672256612323416),
            (1.0, 0.8427007929497149),
            (1.2, 0.9103139782296353),
            (1.25, 0.9229001282564583),
            (1.5, 0.9661051464753108),
            (2.0, 0.9953222650189527),
            (2.5, 0.999593047982555),
            (2.857, 0.9999466417399131),
            (3.0, 0.9999779095030014),
            (4.0, 0.9999999845827421),
            (5.0, 0.9999999999984626),
            (5.9, 0.9999999999999999),
            (6.0, 1.0),
        ];
        for (x, expected) in reference {
            for (sx, se) in [(x, expected), (-x, -expected)] {
                let got = erf(sx);
                assert!(ulps(got, se) <= 1, "erf({sx}) = {got:e}, expected {se:e}");
            }
        }
        assert!(erf(f64::NAN).is_nan());
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn test_p2_9_erf_is_odd_monotone_and_consistent_with_its_derivative() {
        let mut previous = -1.0;
        for i in -6_000..=6_000 {
            let x = i as f64 * 1e-3;
            let y = erf(x);
            assert_eq!(y, -erf(-x), "erf is odd at {x}");
            assert!(y >= previous, "erf is monotone at {x}");
            previous = y;
            // d/dx erf = 2/√π e^{−x²}: central difference with h = 1e-6.
            let h = 1e-6;
            let fd = (erf(x + h) - erf(x - h)) / (2.0 * h);
            let exact = std::f64::consts::FRAC_2_SQRT_PI * exp(-x * x);
            assert!(
                (fd - exact).abs() <= 1e-8,
                "erf' at {x}: fd {fd}, exact {exact}"
            );
        }
    }

    #[test]
    fn test_p2_9_gelu_values_and_derivative() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_344_746_068_542_9).abs() < 1e-15);
        assert!((gelu(-1.0) + 0.158_655_253_931_457_07).abs() < 1e-15);
        for i in -400..=400 {
            let x = i as f64 * 0.01;
            let h = 1e-6;
            let fd = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            assert!((fd - gelu_grad(x)).abs() <= 1e-8, "gelu' at {x}");
        }
    }
}
