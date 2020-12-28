cmake_minimum_required( VERSION 2.8 )

add_executable(HeatSeq 
    main.cxx 
    Heat.cxx 
    HeatSeqIteration.cxx
    HeatSeqDifference.cxx
    Matrix.cxx
)

install(TARGETS HeatSeq DESTINATION .)
