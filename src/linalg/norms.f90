module linalg_norms
  use iso_fortran_env, only: real64
  implicit none
  private

  integer, parameter :: dp = real64

  public :: norm_l1, norm_l2, norm_linf, norm_frobenius

contains

  pure real(dp) function norm_l1(v, n)
    integer,   intent(in) :: n
    real(dp),  intent(in) :: v(n)
    norm_l1 = sum(abs(v))
  end function norm_l1

  pure real(dp) function norm_l2(v, n)
    integer,   intent(in) :: n
    real(dp),  intent(in) :: v(n)
    norm_l2 = sqrt(sum(v**2.0_dp))
  end function norm_l2

  pure real(dp) function norm_linf(v, n)
    integer,   intent(in) :: n
    real(dp),  intent(in) :: v(n)
    norm_linf = maxval(abs(v))
  end function norm_linf

  pure real(dp) function norm_frobenius(A, m, n)
    integer,  intent(in) :: m, n
    real(dp), intent(in) :: A(m, n)
    norm_frobenius = sqrt(sum(A**2.0_dp))
  end function norm_frobenius

end module linalg_norms