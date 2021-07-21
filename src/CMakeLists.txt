set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
aux_source_directory(${CMAKE_SOURCE_DIR}/src FILES_SRC)

pybind11_add_module(dpcxx ${FILES_SRC})

target_link_libraries(dpcxx
PRIVATE
fmt-header-only
)