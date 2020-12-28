#add_definitions("-Wa,-q")

add_library(timer
  timer.cxx
  timer.hxx
  time_unix.c
  time_unix.h
  time_windows.c
  time_windows.h
)
