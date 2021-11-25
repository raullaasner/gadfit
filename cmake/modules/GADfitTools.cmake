# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

# Convert a string into a list. If the variable was already given as
# list, do nothing.
macro(convert_to_list var)
  list(LENGTH ${var} _length)
  if (_length EQUAL 1)
    string(REPLACE " " ";" ${var} ${${var}})
  endif()
endmacro()

# Go through directories listed in LIB_PATHS and turn the entries of
# LIBS into actual targets.
function(generate_library_targets _PATHS _LIBS)
  foreach(LIB ${${_LIBS}})
    # If the target already exists, skip it. This can occur when the
    # same library is set multiple times in LIBS due to circular
    # dependencies.
    if (NOT TARGET ${LIB})
      find_library(LIB_FULLPATH ${LIB} HINTS ${${_PATHS}})
      if (LIB_FULLPATH)
        message(STATUS "Found ${LIB_FULLPATH}")
        add_library(${LIB} UNKNOWN IMPORTED)
        set_target_properties(${LIB} PROPERTIES
          IMPORTED_LOCATION ${LIB_FULLPATH})
        unset(LIB_FULLPATH CACHE)
      else()
        message(FATAL_ERROR "Could not find ${LIB}")
      endif()
    endif()
  endforeach()
endfunction()

# Determine current flags
macro(get_all_flags language flags)
  set(${flags} ${CMAKE_${language}_FLAGS})
  if (CMAKE_BUILD_TYPE)
    string(TOUPPER ${CMAKE_BUILD_TYPE} buildtype)
    set(${flags} "${${flags}} ${CMAKE_${language}_FLAGS_${buildtype}}")
    string(STRIP ${${flags}} ${flags})
  endif()
endmacro()

# Without this macro, the following statements would need to be
# duplicated in the Fortran/gadfit and C++/gadfit folders. It depends
# on the test target, which is defined separately fort Fortran and
# C++, but otherwise the statements are identical for both.
macro(include_coverage)
  if (INCLUDE_COVERAGE)
    find_program(_lcov lcov)
    if (_lcov)
      set(lcov_excluded_directory /usr)
      add_custom_target(coverage
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMAND lcov -c -d CMakeFiles -o cov.info
        COMMAND
        sed -nr 's|SF:${lcov_excluded_directory}|${lcov_excluded_directory}|p'
        cov.info | xargs lcov -r cov.info -o cov_filtered.info
        COMMAND genhtml cov_filtered.info -o out
        COMMAND cmake -E echo
        "Open ${PROJECT_BINARY_DIR}/out/index.html to see the detailed report"
        DEPENDS test)
    else()
      add_custom_target(coverage
        COMMAND cmake -E echo "Error: Cannot run coverage. Install lcov first.")
    endif()
  else()
    add_custom_target(coverage
      COMMAND cmake -E echo
      "Error: Cannot run coverage. Enable INCLUDE_COVERAGE first.")
  endif()
endmacro()
