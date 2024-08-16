set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_for_310p_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_broadcast_plus_broadcast.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)
