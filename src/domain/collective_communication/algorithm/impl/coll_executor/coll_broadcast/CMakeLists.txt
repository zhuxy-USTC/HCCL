set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_comm_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(310P)