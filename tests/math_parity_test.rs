use candle_core::{DType, Device, Tensor};

#[test]
fn test_forward_pass_produces_valid_output() {
    let device = Device::Cpu;
    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states =
        Tensor::randn(0f32, 1f32, (1, 8, 128), &device).unwrap();

    let output = block.forward(&hidden_states, None).unwrap();
    let shape = output.shape();
    assert_eq!(shape.dims(), &[1, 8, 128], "Output shape changed");
}

#[test]
fn test_forward_deterministic_same_block() {
    let device = Device::Cpu;
    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states = Tensor::ones((2, 4, 128), DType::F32, &device).unwrap();

    let run1 = block.forward(&hidden_states, None).unwrap();
    let run2 = block.forward(&hidden_states, None).unwrap();

    let diff = (&run1 - &run2).unwrap();
    let abs_sum: f32 = diff.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(abs_sum < 1e-5, "Non-deterministic output: diff={}", abs_sum);
}

#[test]
fn test_forward_with_causal_mask() {
    let device = Device::Cpu;
    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states =
        Tensor::randn(0f32, 1f32, (1, 8, 128), &device).unwrap();

    let n_heads = block.n_heads;
    let seq_len: usize = hidden_states.dims()[1];
    let mask_vec: Vec<f32> = (0..(n_heads * seq_len * seq_len))
        .map(|i| {
            let _h = i / (seq_len * seq_len);
            let r = (i % (seq_len * seq_len)) / seq_len;
            let c = i % seq_len;
            if c > r { f32::NEG_INFINITY } else { 0.0 }
        })
        .collect();
    let mask = Tensor::from_vec(mask_vec, (n_heads, seq_len, seq_len), &device).unwrap();

    let output = block.forward(&hidden_states, Some(&mask)).unwrap();
    assert_eq!(output.shape().dims(), &[1, 8, 128]);
}

#[test]
fn test_math_parity_across_runs() {
    let device = Device::Cpu;
    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states = Tensor::ones((1, 4, 128), DType::F32, &device).unwrap();

    // Run 5 times, all should produce identical output
    let reference = block.forward(&hidden_states, None).unwrap();
    for _ in 0..5 {
        let output = block.forward(&hidden_states, None).unwrap();
        let diff = (&output - &reference).unwrap();
        let abs_sum: f32 = diff.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(
            abs_sum < 1e-5,
            "Output drift: abs_diff_sum = {}",
            abs_sum
        );
    }
}

#[test]
fn test_output_is_nonzero() {
    let device = Device::Cpu;
    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states = Tensor::ones((1, 8, 128), DType::F32, &device).unwrap();
    let output = block.forward(&hidden_states, None).unwrap();
    let out_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    // At least some values should be non-zero
    let nonzero_count = out_vec.iter().filter(|&&f| f != 0.0).count();
    assert!(nonzero_count > 0, "All output values are zero");
}
