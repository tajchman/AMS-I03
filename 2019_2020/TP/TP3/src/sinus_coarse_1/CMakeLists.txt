get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_definitions(
    ${OpenMP_CXX_FLAGS} 
   )

add_executable(${CODE} 
  sinus.cxx
  ../sin.cxx
)

target_link_libraries(${CODE}
  util
  ${OpenMP_CXX_LIBRARIES}
)

install(TARGETS ${CODE} DESTINATION .)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()


