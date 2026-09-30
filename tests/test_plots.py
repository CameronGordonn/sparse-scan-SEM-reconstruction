from semrecon.plots import frontier, frontier_table, matched


def _row(method, image, pattern="uniform", frac=0.1, regime="fixed_dwell"):
    return {"method": method, "image": image, "pattern": pattern, "frac": frac, "regime": regime}


def test_matched_keeps_only_shared_images():
    rows = [_row("tv_l2", i) for i in range(4)] + [_row("diffusion", i) for i in range(2)]
    kept, dropped = matched(rows)
    assert dropped == 2
    assert {(r["method"], r["image"]) for r in kept} == {(m, i) for m in ("tv_l2", "diffusion") for i in (0, 1)}


def test_matched_is_per_case():
    # a method missing from one case must not restrict the images of another case
    rows = [_row("tv_l2", i) for i in range(3)] + [_row("diffusion", 0)]
    rows += [_row("tv_l2", i, pattern="line_hop") for i in range(3)]
    kept, dropped = matched(rows)
    assert dropped == 2
    assert sum(r["pattern"] == "line_hop" for r in kept) == 3


def test_frontier_keeps_only_improvements_in_time_order():
    def r(pattern, frac, t, psnr):
        return {"method": "unet", "image": 0, "pattern": pattern, "frac": frac, "regime": "fixed_dwell",
                "psnr": psnr, "scan_time_rel": t}

    rows = [r("partial_raster", 0.1, 0.101, 24.0), r("partial_raster", 0.3, 0.301, 26.0),
            r("uniform", 0.05, 0.34, 25.0),  # slower and worse than raster 30%: not on the frontier
            r("uniform", 0.3, 1.31, 26.5)]
    pts = frontier(rows, "fixed_dwell")["unet"]
    assert [(p[2], p[3]) for p in pts] == [("partial_raster", 0.1), ("partial_raster", 0.3), ("uniform", 0.3)]
    table = frontier_table(rows, "fixed_dwell", budgets=(0.1, 0.3, 1.0))
    assert "24.00 (raster 10%" in table and "26.00 (raster 30%" in table  # 1% budget slack


def test_coil_error_table_reports_paired_drops():
    from semrecon.plots import coil_error_table

    rows = [{"method": "unet", "pattern": "uniform", "frac": 0.1, "image": str(i), "amp": a,
             "psnr": 25.0 + i - 2 * a} for i in range(3) for a in (0.0, 1.0)]
    table = coil_error_table(rows)
    assert "| 26.00 | -2.00 |" in table and "3 test images" in table
