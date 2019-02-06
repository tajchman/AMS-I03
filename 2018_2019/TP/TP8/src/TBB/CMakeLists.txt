cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(TBB)

include_directories(
  ${TBB_INCLUDE_DIRS}
  )

foreach(d 0 1)
  add_executable(HeatTBB_${d}
    main.cxx
    Heat.cxx
    HeatTBBIteration_${d}.cxx
    HeatTBBDifference_${d}.cxx
    Matrix.cxx
)

  target_link_libraries(HeatTBB_${d} ${TBB_LIBRARIES})

  install(TARGETS HeatTBB_${d} DESTINATION .)

endforeach()


