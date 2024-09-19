set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_operator_for_hetero.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/send_receive_operator.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)
