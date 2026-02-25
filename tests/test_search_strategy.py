import pytest
from nas.search_strategy.random_search import RandomSearch


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_space(n: int = 20) -> list:
    """Dummy search space of n architecture placeholders."""
    return [{"id": i} for i in range(n)]


def _make_config(iters: int = 5) -> dict:
    return {"SearchStrategy": {"type": "RandomSearch", "nbr_iterations": iters}}


# ─── Constructor ────────────────────────────────────────────────────────────

class TestRandomSearchInit:

    def test_nbr_iterations_set(self):
        rs = RandomSearch(_make_space(), _make_config(iters=7))
        assert rs.nbr_iterations == 7

    def test_search_space_stored(self):
        space = _make_space(10)
        rs = RandomSearch(space, _make_config())
        assert rs.search_space is space

    def test_raises_on_empty_space(self):
        with pytest.raises(ValueError, match="empty"):
            RandomSearch([], _make_config())


# ─── search() output contract ────────────────────────────────────────────────

class TestRandomSearchOutput:

    def test_returns_list(self):
        rs = RandomSearch(_make_space(), _make_config(iters=3))
        assert isinstance(rs.search(), list)

    def test_returns_correct_number_of_indices(self):
        iters = 4
        rs = RandomSearch(_make_space(20), _make_config(iters=iters))
        result = rs.search()
        assert len(result) == iters

    def test_all_indices_in_valid_range(self):
        space = _make_space(15)
        rs = RandomSearch(space, _make_config(iters=10))
        for idx in rs.search():
            assert 0 <= idx < len(space)

    def test_no_duplicate_when_space_larger_than_iters(self):
        """When iters < space size, sampled indices should be unique."""
        rs = RandomSearch(_make_space(50), _make_config(iters=10))
        result = rs.search()
        assert len(result) == len(set(result))

    def test_repeated_calls_not_deterministic(self):
        """Two independent calls should (with overwhelming probability) differ."""
        rs1 = RandomSearch(_make_space(100), _make_config(iters=20))
        rs2 = RandomSearch(_make_space(100), _make_config(iters=20))
        # Extremely unlikely to collide — 1 in 100^20
        assert rs1.search() != rs2.search()

    def test_works_when_iters_equals_space_size(self):
        size = 5
        rs = RandomSearch(_make_space(size), _make_config(iters=size))
        result = rs.search()
        assert len(result) == size
        assert len(set(result)) == size  # all unique

    def test_works_when_iters_exceeds_space_size(self):
        """Should still return nbr_iterations indices (with repeats allowed)."""
        rs = RandomSearch(_make_space(3), _make_config(iters=5))
        result = rs.search()
        assert len(result) == 5
