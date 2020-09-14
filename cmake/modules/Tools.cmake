# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

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
