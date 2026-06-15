@0xd1e7a8c3f5b9e012;

struct ForwardPassRequest {
    route    @0 :List(Data);
    hopIndex @1 :UInt32;
    requestId @2 :UInt64;
    tensor   @3 :Tensor;
    mask     @4 :Tensor;
}

struct ForwardPassResponse {
    requestId @0 :UInt64;
    tensor   @1 :Tensor;
}

struct InferenceSessionRequest {
    sessionId     @0 :Text;
    stepId        @1 :Text;
    stepCount     @2 :UInt64;
    maxNewTokens  @3 :UInt32;
    positionIds   @4 :List(UInt64);
    inputs        @5 :Tensor;
}

struct InferenceSessionResponse {
    sessionId @0 :Text;
    stepId    @1 :Text;
    outputs   @2 :Tensor;
}

struct Tensor {
    dtype @0 :Dtype;
    shape @1 :List(UInt64);
    data  @2 :Data;
}

enum Dtype {
    f16 @0;
    bf16 @1;
    f32 @2;
}

