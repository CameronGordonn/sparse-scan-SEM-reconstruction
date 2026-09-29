from semrecon.plots import matched


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
