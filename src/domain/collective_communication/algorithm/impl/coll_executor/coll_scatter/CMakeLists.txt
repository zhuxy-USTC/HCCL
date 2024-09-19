set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_single_rank_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

