# - Find kokkos
#
#  KOKKOS_INCLUDE_DIRS  - where to find kokkos.h, etc.
#  KOKKOS_LIBRARIES     - List of libraries when using kokkos.
#  KOKKOS_FOUND         - True if kokkos found.
#

find_path (KOKKOS_ROOT_DIR
  NAMES include/Kokkos_Core.hpp
  PATHS ENV KOKKOS_ROOT
  DOC "Kokkos root directory")

message(STATUS "Kokkos root directory ${KOKKOS_ROOT_DIR}")

FIND_PATH (KOKKOS_INCLUDE_DIR
  NAMES Kokkos_Core.hpp
  HINTS ${KOKKOS_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "Kokkos include directory")

find_library(KOKKOS_LIBRARY NAMES kokkos kokkoscore HINTS ${KOKKOS_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KOKKOS DEFAULT_MSG KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY)

mark_as_advanced(KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY)

