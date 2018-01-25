#add_definitions("-Wa,-q")

add_library(timer
  timer.cxx
  time_unix.c
  time_windows.c
)
