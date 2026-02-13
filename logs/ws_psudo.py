if is_dma_warp_group:
    cute.arch.warpgroup_reg_dealloc(40)
    producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, self.ab_stage
    )
    for k_tile in range(k_tile_cnt):
        mainloop_pipeline.producer_acquire(producer_state)
        copy()  # TMA
        mainloop_pipeline.producer_commit(producer_state)
        producer_state.advance()
if is_mma_warp_group:
    cute.arch.warpgroup_reg_alloc(232)
    consumer_read_state = pipeline.make_pipeline_state(
         pipeline.PipelineUserType.Consumer, self.ab_stage
    )
    consumer_release_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, self.ab_stage
    )
    for k_tile in range(k_tile_cnt):
        mainloop_pipeline.consumer_wait(consumer_read_state)
        gemm()  # MMA
        mainloop_pipeline.consumer_release(consumer_release_state)
        consumer_read_state.advance()
        consumer_release_state.advance()
# Epilogue
