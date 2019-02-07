cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(TBB)

include_directories(
  ${TBB_INCLUDE_DIRS}
  )

foreach(d 0 1 3)
  add_executable(HeatTBB_${d}
    main.cxx
    Heat.cxx
    HeatTBBIteration_${d}.cxx
    HeatTBBDifference_${d}.cxx
    Matrix.cxx
)
endforeach()

add_executable(HeatTBB_2
  main.cxx
  Heat.cxx
  HeatTBBIteration_1.cxx
  HeatTBBDifference_1.cxx
  MatrixTBB.cxx
)

set_target_properties(HeatTBB_3  PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
foreach(d 0 1 2 3)
  target_link_libraries(HeatTBB_${d} ${TBB_LIBRARIES})

  install(TARGETS HeatTBB_${d} DESTINATION .)

endforeach()


