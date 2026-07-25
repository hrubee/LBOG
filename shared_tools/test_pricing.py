"""Tests for pricing.py — Black-Scholes option pricing and Greeks."""

import math
import pytest

from pricing import norm_cdf, norm_pdf, bs_price, bs_greeks, bs_price_and_greeks


# ─── norm_cdf ──────────────────────────────────

class TestNormCdf:
    def test_zero(self):
        assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_large_positive(self):
        # N(4) ≈ 0.99997
        assert norm_cdf(4.0) == pytest.approx(0.9999683, abs=1e-5)

    def test_large_negative(self):
        # N(-4) ≈ 0.00003
        assert norm_cdf(-4.0) == pytest.approx(1.0 - 0.9999683, abs=1e-5)

    def test_symmetry(self):
        # N(x) + N(-x) = 1
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-12)

    def test_known_values(self):
        # N(1) ≈ 0.8413, N(-1) ≈ 0.1587
        assert norm_cdf(1.0) == pytest.approx(0.8413, abs=1e-4)
        assert norm_cdf(-1.0) == pytest.approx(0.1587, abs=1e-4)


# ─── norm_pdf ──────────────────────────────────

class TestNormPdf:
    def test_zero(self):
        # phi(0) = 1/sqrt(2*pi) ≈ 0.3989
        assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-10)

    def test_symmetry(self):
        # phi(x) = phi(-x)
        for x in [0.5, 1.0, 2.0]:
            assert norm_pdf(x) == pytest.approx(norm_pdf(-x), abs=1e-12)

    def test_decreasing(self):
        # pdf decreases as |x| increases
        assert norm_pdf(0.0) > norm_pdf(1.0) > norm_pdf(2.0) > norm_pdf(3.0)

    def test_known_value(self):
        # phi(1) = exp(-0.5) / sqrt(2*pi) ≈ 0.2420
        assert norm_pdf(1.0) == pytest.approx(0.2420, abs=1e-4)


# ─── bs_price ──────────────────────────────────

class TestBsPrice:
    def test_atm_call_known_value(self):
        # ATM call: S=K=100, T=1yr, vol=20%, r=5%
        # Known BS price ≈ $10.45
        price = bs_price(100, 100, 365, 0.20, risk_free=0.05, option_type="call")
        assert price == pytest.approx(10.45, abs=0.1)

    def test_atm_put_known_value(self):
        # ATM put: S=K=100, T=1yr, vol=20%, r=5%
        # Known BS price ≈ $5.57
        price = bs_price(100, 100, 365, 0.20, risk_free=0.05, option_type="put")
        assert price == pytest.approx(5.57, abs=0.1)

    def test_put_call_parity(self):
        # C - P = S - K*exp(-rT)
        S, K, dte, vol, r = 100, 100, 365, 0.30, 0.05
        call = bs_price(S, K, dte, vol, r, "call")
        put = bs_price(S, K, dte, vol, r, "put")
        T = dte / 365.0
        expected_diff = S - K * math.exp(-r * T)
        assert (call - put) == pytest.approx(expected_diff, abs=1e-8)

    def test_deep_itm_call(self):
        # Deep ITM call: S=150, K=100, high vol — price > intrinsic
        price = bs_price(150, 100, 30, 0.50, 0.05, "call")
        intrinsic = 150 - 100
        assert price >= intrinsic

    def test_deep_otm_call(self):
        # Deep OTM call: S=50, K=100 — price near zero
        price = bs_price(50, 100, 30, 0.20, 0.05, "call")
        assert price < 1.0
        assert price >= 0.0

    def test_expired_call_itm(self):
        # dte=0, ITM call → intrinsic value
        price = bs_price(110, 100, 0, 0.30, 0.05, "call")
        assert price == pytest.approx(10.0, abs=1e-10)

    def test_expired_put_otm(self):
        # dte=0, OTM put → 0
        price = bs_price(110, 100, 0, 0.30, 0.05, "put")
        assert price == pytest.approx(0.0, abs=1e-10)

    def test_expired_put_itm(self):
        # dte=0, ITM put → intrinsic
        price = bs_price(90, 100, 0, 0.30, 0.05, "put")
        assert price == pytest.approx(10.0, abs=1e-10)

    def test_zero_vol_itm_call(self):
        # vol=0, ITM → intrinsic
        price = bs_price(110, 100, 30, 0, 0.05, "call")
        assert price == pytest.approx(10.0, abs=1e-10)

    def test_zero_vol_otm_call(self):
        # vol=0, OTM → 0
        price = bs_price(90, 100, 30, 0, 0.05, "call")
        assert price == pytest.approx(0.0, abs=1e-10)

    def test_higher_vol_higher_price(self):
        # Higher vol → higher option price (all else equal)
        low_vol = bs_price(100, 100, 30, 0.20, 0.05, "call")
        high_vol = bs_price(100, 100, 30, 0.50, 0.05, "call")
        assert high_vol > low_vol

    def test_longer_dte_higher_price(self):
        # Longer DTE → higher call price (all else equal)
        short = bs_price(100, 100, 7, 0.30, 0.05, "call")
        long = bs_price(100, 100, 90, 0.30, 0.05, "call")
        assert long > short

    def test_btc_atm_call(self):
        # Realistic BTC: S=K=95000, 30d, 80% vol
        price = bs_price(95000, 95000, 30, 0.80, 0.05)
        # Should be a significant premium for 80% vol
        assert price > 5000
        assert price < 20000


# ─── bs_greeks ─────────────────────────────────

class TestBsGreeks:
    def test_atm_call_delta_near_half(self):
        # ATM call delta ≈ 0.5 (slightly above due to risk-free rate)
        g = bs_greeks(100, 100, 365, 0.20, 0.05, "call")
        assert g["delta"] == pytest.approx(0.6368, abs=0.02)

    def test_atm_put_delta_near_neg_half(self):
        g = bs_greeks(100, 100, 365, 0.20, 0.05, "put")
        assert g["delta"] == pytest.approx(-0.3632, abs=0.02)

    def test_call_delta_range(self):
        # Call delta ∈ [0, 1]
        g = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert 0 <= g["delta"] <= 1

    def test_put_delta_range(self):
        # Put delta ∈ [-1, 0]
        g = bs_greeks(100, 100, 30, 0.30, 0.05, "put")
        assert -1 <= g["delta"] <= 0

    def test_gamma_positive(self):
        g = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert g["gamma"] > 0

    def test_gamma_same_for_call_and_put(self):
        gc = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        gp = bs_greeks(100, 100, 30, 0.30, 0.05, "put")
        assert gc["gamma"] == pytest.approx(gp["gamma"], abs=1e-6)

    def test_vega_positive(self):
        g = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert g["vega"] > 0

    def test_vega_same_for_call_and_put(self):
        gc = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        gp = bs_greeks(100, 100, 30, 0.30, 0.05, "put")
        assert gc["vega"] == pytest.approx(gp["vega"], abs=1e-6)

    def test_theta_negative_for_long_options(self):
        # Time decay: theta < 0 for calls and puts
        gc = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        gp = bs_greeks(100, 100, 30, 0.30, 0.05, "put")
        assert gc["theta"] < 0
        assert gp["theta"] < 0

    def test_expired_returns_zeros(self):
        g = bs_greeks(100, 100, 0, 0.30, 0.05, "call")
        assert g["delta"] == 0.0
        assert g["gamma"] == 0.0
        assert g["theta"] == 0.0
        assert g["vega"] == 0.0

    def test_zero_spot_returns_zeros(self):
        g = bs_greeks(0, 100, 30, 0.30, 0.05, "call")
        assert g["delta"] == 0.0

    def test_deep_itm_call_delta_near_one(self):
        g = bs_greeks(200, 100, 30, 0.20, 0.05, "call")
        assert g["delta"] == pytest.approx(1.0, abs=0.01)

    def test_deep_otm_call_delta_near_zero(self):
        g = bs_greeks(50, 100, 30, 0.20, 0.05, "call")
        assert g["delta"] == pytest.approx(0.0, abs=0.01)


# ─── bs_price_and_greeks ──────────────────────

class TestBsPriceAndGreeks:
    def test_returns_tuple(self):
        result = bs_price_and_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_price_matches_standalone(self):
        price_standalone = bs_price(100, 100, 30, 0.30, 0.05, "call")
        price_combined, _ = bs_price_and_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert price_combined == pytest.approx(price_standalone, abs=1e-10)

    def test_greeks_match_standalone(self):
        greeks_standalone = bs_greeks(100, 100, 30, 0.30, 0.05, "call")
        _, greeks_combined = bs_price_and_greeks(100, 100, 30, 0.30, 0.05, "call")
        for key in ("delta", "gamma", "theta", "vega"):
            assert greeks_combined[key] == pytest.approx(greeks_standalone[key], abs=1e-10)

    def test_greeks_dict_keys(self):
        _, greeks = bs_price_and_greeks(100, 100, 30, 0.30, 0.05, "call")
        assert set(greeks.keys()) == {"delta", "gamma", "theta", "vega"}
