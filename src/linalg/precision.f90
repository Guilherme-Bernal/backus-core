module precision
  implicit none
  public :: dp
  integer, parameter :: dp = kind(1.0d0)
end module precision