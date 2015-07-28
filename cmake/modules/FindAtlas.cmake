# Copyright (C) 2014-2015 Raul Laasner
# This file is distributed under the terms of the GNU General Public
# License, see LICENSE in the root directory of the present
# distribution or http://gnu.org/copyleft/gpl.txt .

include(FindPackageHandleStandardArgs)

find_path(Atlas_INCLUDE_DIR cblas.h /usr/include /usr/local/include)
find_package_handle_standard_args(Atlas DEFAULT_MSG Atlas_INCLUDE_DIR)
mark_as_advanced(Atlas_INCLUDE_DIR)

find_library(Atlas_LIBRARY NAMES tatlas PATHS /usr/lib /usr/local/lib)
find_package_handle_standard_args(Atlas DEFAULT_MSG Atlas_LIBRARY)
mark_as_advanced(Atlas_LIBRARY)
set(Atlas_LIBRARIES ${Atlas_LIBRARY})
