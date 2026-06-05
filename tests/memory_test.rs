use candle_core::{Device, Tensor};
use std::sync::atomic::{AtomicI64, Ordering};

static ALLOC_COUNT: AtomicI64 = AtomicI64::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct CountingAlloc;

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(layout.size() as i64, Ordering::SeqCst);
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        ALLOC_COUNT.fetch_sub(layout.size() as i64, Ordering::SeqCst);
        std::alloc::System.dealloc(ptr, layout)
    }
}

#[test]
fn test_memory_stability_100_passes() {
    let device = Device::Cpu;

    let block = inferencemesh_compute::TransformerBlock::create_test_block(&device).unwrap();

    let hidden_states =
        Tensor::randn(0f32, 1f32, (1, 8, 128), &device).unwrap();

    ALLOC_COUNT.store(0, Ordering::SeqCst);

    for _ in 0..10 {
        let _output = block.forward(&hidden_states, None).unwrap();
    }

    let base = ALLOC_COUNT.load(Ordering::SeqCst);

    for _ in 0..90 {
        let _output = block.forward(&hidden_states, None).unwrap();
    }

    let final_net = ALLOC_COUNT.load(Ordering::SeqCst);
    let growth = final_net - base;

    println!("Allocation growth over 90 additional passes: {} bytes", growth);

    assert!(
        growth < 10_000_000,
        "Memory leak detected: allocation growth = {}",
        growth
    );
}
