program power2

  implicit none

  integer :: argc
  integer i, j, n, nargs
  double precision, dimension(:), allocatable :: v, w
  double precision, dimension(:,:), allocatable :: a
  double precision :: normalise, variation
  
  character(len=128) BUFFER
  double precision :: s, lambda, lambda0
  integer :: k, kmax
  
  n = 3000
  nargs = iargc()
  if (nargs .gt. 0) then
     call getarg(1, BUFFER)
     read(BUFFER, *) n
  endif
  write (*,*) 'n = ', n
  allocate(a(n,n), v(n), w(n))

  call init(a, v, n)

  lambda = 0.0
  kmax = 100

  do k=1,kmax

     lambda0 = lambda

     call produit_matrice_vecteur(w, a, v, n)
     
     lambda = normalise(w, n)
     v = w

     call affiche(k, lambda)
     if (variation(lambda,lambda0) < 1.0D-12) exit
  enddo
    
  deallocate(a,v,w)
  write (*,*)
  
end program power2

subroutine produit_matrice_vecteur(w, a, v, n)
  implicit none

  integer :: i, j, n
  double precision :: w(n), a(n,n), v(n), s

  w = 0
  do i=1,n
     s = 0.0d0
     do j=1,n
        s = s + a(i,j)*v(j)
     enddo
     w(i) = s
  enddo
  
end subroutine produit_matrice_vecteur

subroutine init(a, v, n)
  implicit none

  integer :: i, n
  double precision :: a(n,n), v(n), s
  double precision :: normalise
  
  do i=1,n
     call random_number(v(i))
  enddo

  s = normalise(v, n)

  a = 1.0d0/n
  do i=1, n
     a(i,i) = 5.0d0 + 1.0d0/n
  enddo

end subroutine init


double precision function normalise(v, n)
  implicit none

  integer :: i, n
  double precision :: s, v(n)

  s = 0.0
  do i=1,n
     s = s + v(i)*v(i)
  enddo

  s = sqrt(s)
  do i=1,n
     v(i) = v(i)/s
  enddo

  normalise = s
end function normalise

double precision function variation(u, v)
  implicit none
  double precision :: u, v

  variation = abs(u-v)/(abs(u) + abs(v) + 1.0D0);
end function variation
