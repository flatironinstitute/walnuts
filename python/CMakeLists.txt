add_library(walnutpie SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/walnuts/walnutpie.cpp")
target_link_libraries(walnutpie PRIVATE walnuts BridgeStan::BridgeStan Eigen3::Eigen)

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
generate_export_header(walnutpie
  EXPORT_FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/src/walnuts/export.h"
)

install(TARGETS walnutpie
  LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/walnuts"
  RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/walnuts"
)
