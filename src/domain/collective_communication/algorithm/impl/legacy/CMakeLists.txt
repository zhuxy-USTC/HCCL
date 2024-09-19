set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_impl.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(operator)
