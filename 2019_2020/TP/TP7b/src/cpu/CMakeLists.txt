add_executable(eqn_cpu
  main.cxx
  calcul.cxx
  laplacien.cxx
  init.cxx
  forces.cxx
  variation.cxx
  save.cxx
  )

install(TARGETS eqn_cpu DESTINATION .)



