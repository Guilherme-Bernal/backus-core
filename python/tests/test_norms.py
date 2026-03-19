import pytest
import numpy as np
from backus.linalg import norm_l1, norm_l2, norm_linf, norm_frobenius

TOLERANCE = 1e-10
@pytest.fixture
def simple_vector():
    return np.array([3.0, 4.0, 0.0, -5.0, 2.0])


@pytest.fixture
def unit_vector():
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def negative_vector():
    return np.array([-1.0, -2.0, -3.0, -4.0])


@pytest.fixture
def simple_matrix():
    return np.array([[1.0, 2.0],
                     [3.0, 4.0]])


@pytest.fixture
def large_vector():
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(10_000)


@pytest.fixture
def large_matrix():
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((200, 200))

class TestNormL1:

    def test_simple_vector(self, simple_vector):
        result   = norm_l1(simple_vector)
        expected = np.linalg.norm(simple_vector, ord=1)
        assert abs(result - expected) < TOLERANCE

    def test_unit_vector(self, unit_vector):
        assert abs(norm_l1(unit_vector) - 1.0) < TOLERANCE

    def test_negative_vector(self, negative_vector):
        result   = norm_l1(negative_vector)
        expected = np.linalg.norm(negative_vector, ord=1)
        assert abs(result - expected) < TOLERANCE

    def test_single_element(self):
        assert abs(norm_l1([-7.5]) - 7.5) < TOLERANCE

    def test_large_vector(self, large_vector):
        result   = norm_l1(large_vector)
        expected = np.linalg.norm(large_vector, ord=1)
        assert abs(result - expected) < TOLERANCE

    def test_list_input(self):
        result   = norm_l1([1.0, 2.0, 3.0])
        expected = np.linalg.norm([1.0, 2.0, 3.0], ord=1)
        assert abs(result - expected) < TOLERANCE

    def test_returns_float(self, simple_vector):
        assert isinstance(norm_l1(simple_vector), float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            norm_l1([])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            norm_l1([[1.0, 2.0], [3.0, 4.0]])

class TestNormL2:

    def test_pythagorean_triple(self):
        assert abs(norm_l2([3.0, 4.0]) - 5.0) < TOLERANCE

    def test_simple_vector(self, simple_vector):
        result   = norm_l2(simple_vector)
        expected = np.linalg.norm(simple_vector)
        assert abs(result - expected) < TOLERANCE

    def test_unit_vector(self, unit_vector):
        assert abs(norm_l2(unit_vector) - 1.0) < TOLERANCE

    def test_negative_vector(self, negative_vector):
        result   = norm_l2(negative_vector)
        expected = np.linalg.norm(negative_vector)
        assert abs(result - expected) < TOLERANCE

    def test_large_vector(self, large_vector):
        result   = norm_l2(large_vector)
        expected = np.linalg.norm(large_vector)
        assert abs(result - expected) < TOLERANCE

    def test_list_input(self):
        result   = norm_l2([1.0, 0.0, 0.0])
        assert abs(result - 1.0) < TOLERANCE

    def test_returns_float(self, simple_vector):
        assert isinstance(norm_l2(simple_vector), float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            norm_l2([])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            norm_l2([[1.0, 2.0]])

class TestNormLinf:

    def test_simple_vector(self, simple_vector):
        result   = norm_linf(simple_vector)
        expected = np.linalg.norm(simple_vector, ord=np.inf)
        assert abs(result - expected) < TOLERANCE

    def test_negative_vector(self, negative_vector):
        result   = norm_linf(negative_vector)
        expected = np.linalg.norm(negative_vector, ord=np.inf)
        assert abs(result - expected) < TOLERANCE

    def test_max_is_negative(self):
        v = [-10.0, 1.0, 2.0]
        assert abs(norm_linf(v) - 10.0) < TOLERANCE

    def test_unit_vector(self, unit_vector):
        assert abs(norm_linf(unit_vector) - 1.0) < TOLERANCE

    def test_single_element(self):
        assert abs(norm_linf([-3.0]) - 3.0) < TOLERANCE

    def test_large_vector(self, large_vector):
        result   = norm_linf(large_vector)
        expected = np.linalg.norm(large_vector, ord=np.inf)
        assert abs(result - expected) < TOLERANCE

    def test_returns_float(self, simple_vector):
        assert isinstance(norm_linf(simple_vector), float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            norm_linf([])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            norm_linf([[1.0, 2.0]])
            
class TestNormFrobenius:

    def test_simple_matrix(self, simple_matrix):
        result   = norm_frobenius(simple_matrix)
        expected = np.linalg.norm(simple_matrix, ord='fro')
        assert abs(result - expected) < TOLERANCE

    def test_identity_matrix(self):
        I = np.eye(4)
        assert abs(norm_frobenius(I) - 2.0) < TOLERANCE

    def test_zero_matrix(self):
        Z = np.zeros((3, 3))
        assert abs(norm_frobenius(Z) - 0.0) < TOLERANCE

    def test_single_element(self):
        assert abs(norm_frobenius([[5.0]]) - 5.0) < TOLERANCE

    def test_large_matrix(self, large_matrix):
        result   = norm_frobenius(large_matrix)
        expected = np.linalg.norm(large_matrix, ord='fro')
        assert abs(result - expected) < TOLERANCE

    def test_non_square_matrix(self):
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        result   = norm_frobenius(A)
        expected = np.linalg.norm(A, ord='fro')
        assert abs(result - expected) < TOLERANCE

    def test_list_input(self):
        result   = norm_frobenius([[3.0, 4.0]])
        assert abs(result - 5.0) < TOLERANCE

    def test_returns_float(self, simple_matrix):
        assert isinstance(norm_frobenius(simple_matrix), float)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            norm_frobenius(np.zeros((0, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            norm_frobenius([1.0, 2.0, 3.0])