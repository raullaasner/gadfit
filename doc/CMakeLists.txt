# This Source Code Form is subject to the terms of the GNU General
# Public License, v. 3.0. If a copy of the GPL was not distributed
# with this file, You can obtain one at
# http://gnu.org/copyleft/gpl.txt.

find_package(LATEX)
if (PDFLATEX_COMPILER AND BIBTEX_COMPILER)
  set(SRC ${CMAKE_CURRENT_SOURCE_DIR})
  set(BLD ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(doc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/user_guide.tex ${BLD}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/refs.bib       ${BLD}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/physrev.bst    ${BLD}
    COMMAND ${PDFLATEX_COMPILER} user_guide.tex
    COMMAND ${BIBTEX_COMPILER}   user_guide.aux
    COMMAND ${PDFLATEX_COMPILER} user_guide.tex
    COMMAND ${PDFLATEX_COMPILER} user_guide.tex)
  install(FILES ${BLD}/user_guide.pdf DESTINATION share/${GADFIT} OPTIONAL)
elseif (NOT PDFLATEX_COMPILER)
  message(STATUS "pdflatex not found.")
elseif (NOT BIBTEX_COMPILER)
  message(STATUS "bibtex not found.")
endif()
