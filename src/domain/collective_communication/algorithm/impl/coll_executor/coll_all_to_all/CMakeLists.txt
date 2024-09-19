set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_2level_pipeline_excecutor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_fullmesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_staged_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_single_rank_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_mesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_direct_fullmesh_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)