
# NB: the 'misspelling' as walnutPY not PIE is intentional (and not user-facing)
add_library(walnutpy SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/walnutpie/walnutpy.cpp")
target_link_libraries(walnutpy PRIVATE walnutpie BridgeStan::BridgeStan Eigen3::Eigen)

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
generate_export_header(walnutpy
  EXPORT_FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/src/walnutpie/export.h"
)

install(TARGETS walnutpy
  LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/walnutpie"
  RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/walnutpie"
)
