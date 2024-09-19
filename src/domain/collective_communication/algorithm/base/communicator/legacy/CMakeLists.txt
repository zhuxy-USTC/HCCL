set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_star.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)
