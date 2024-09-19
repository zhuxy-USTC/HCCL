set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_info_extractor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_hd_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_partial_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ring_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_transport_req_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_p2p_transport_req.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(legacy)