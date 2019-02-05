cmake_minimum_required( VERSION 2.8 )

add_executable(heatSeq main.cxx heatSeq.cxx save.cxx)

install(TARGETS heatSeq DESTINATION .)
