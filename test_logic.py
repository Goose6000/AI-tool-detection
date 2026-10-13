from logic import Det, Inventory, Stabilizer, battery_status, CLASS_NAMES


def test_battery_installed():
    drill = Det("m18_hammer_drill", 0.9, (100, 100, 500, 400))
    bat = Det("m18_battery_cp2", 0.9, (200, 300, 350, 420))  # mostly inside drill box
    s = battery_status([drill, bat])
    assert s == {"batteries_total": 1, "installed": 1, "loose": 0}


def test_battery_loose():
    drill = Det("m18_hammer_drill", 0.9, (100, 100, 500, 400))
    bat = Det("m18_battery_cp2", 0.9, (600, 300, 750, 420))
    s = battery_status([drill, bat])
    assert s == {"batteries_total": 1, "installed": 0, "loose": 1}


def test_one_battery_per_drill():
    drill = Det("m18_hammer_drill", 0.9, (100, 100, 500, 400))
    b1 = Det("m18_battery_cp2", 0.9, (200, 300, 350, 420))
    b2 = Det("m18_battery_cp2", 0.9, (220, 310, 360, 420))
    s = battery_status([drill, b1, b2])
    assert s["installed"] == 1 and s["loose"] == 1


def test_removed_and_returned():
    inv = Inventory()
    full = set(CLASS_NAMES)
    assert inv.update(full) == []
    assert inv.update(full - {"hammer"}) == [("REMOVED", "hammer")]
    assert inv.missing() == ["hammer"]
    assert inv.update(full) == [("RETURNED", "hammer")]


def test_stabilizer_ignores_single_missed_frame():
    st = Stabilizer(window=10, need=5)
    for _ in range(10):
        st.update(["pliers"])
    assert "pliers" in st.update([])  # one missed frame
    for _ in range(8):
        present = st.update([])
    assert "pliers" not in present
