use std::sync::{atomic::AtomicUsize, Arc};
use parking_lot::RawRwLock;
use rand::{Rng, RngCore, SeedableRng};
use crate::{utils::State, LocklessSlotmap, SlotmapTicket};

const TEST_SEED: u64 = 0xdead_beef_cafe_babe;
const TEST_ITERATIONS: usize = 512;
const TEST_ITERATIONS_SMALL: usize = 256;
const TEST_ITERATIONS_TINY: usize = 16;
const TEST_ITERATIONS_LARGE: usize = 4096;
const TEST_ITERATIONS_VERY_LARGE: usize = 16384;
const THREAD_COUNT: usize = 8;

///
/// A simple leak detector that increments a counter when created and decrements it when dropped.
/// This is used to detect memory leaks in tests.
///
struct LeakDetector {
    count: Arc<AtomicUsize>,
}

impl LeakDetector {
    fn new(count: Arc<AtomicUsize>) -> Self {
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LeakDetector { count }
    }
}

impl Drop for LeakDetector {
    fn drop(&mut self) {
        self.count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}


#[test]
pub fn test_zero() {
    assert_eq!(0, 0);
}

#[test]
pub fn test_ticket_valid() {
    let ticket = SlotmapTicket::new(0x1234, 0x5678_9abc, 0xdef0);
    assert_eq!(ticket.block_index(), 0x1234, "Block index mismatch");
    assert_eq!(ticket.slot_index(), 0x5678_9abc, "Slot index mismatch");
    assert_eq!(ticket.generation(), 0xdef0, "Generation mismatch");

    let ticket = SlotmapTicket::new(0xffff, 0xffff_ffff, 0xffff);
    assert_eq!(ticket.block_index(), 0xffff, "Block index mismatch");
    assert_eq!(ticket.slot_index(), 0xffff_ffff, "Slot index mismatch");
    assert_eq!(ticket.generation(), 0xffff, "Generation mismatch");

    let ticket = SlotmapTicket::new(0, 0, 0);
    assert_eq!(ticket.block_index(), 0, "Block index mismatch");
    assert_eq!(ticket.slot_index(), 0, "Slot index mismatch");
    assert_eq!(ticket.generation(), 0, "Generation mismatch");

    let ticket = SlotmapTicket::new(0, 0, 0xffff);
    assert_eq!(ticket.block_index(), 0, "Block index mismatch");
    assert_eq!(ticket.slot_index(), 0, "Slot index mismatch");
    assert_eq!(ticket.generation(), 0xffff, "Generation mismatch");
}

#[test]
pub fn test_state_from_and_into() {
    let state = State::Free {
        next_free_slot: Some(0x1234),
        next_generation: 0x5678,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Free {
        next_free_slot: None,
        next_generation: 0x5678,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Occupied {
        generation: 0x5678,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Reserved;
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Free {
        next_free_slot: Some(0xffff),
        next_generation: 0xffff,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Free {
        next_free_slot: None,
        next_generation: 0xffff,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Occupied {
        generation: 0xffff,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Free { 
        next_free_slot: Some(0x0),
        next_generation: 0x0,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);

    let state = State::Free {
        next_free_slot: None,
        next_generation: 0x0,
    };
    let state_u64: u64 = state.into();
    assert_eq!(State::from(state_u64), state);
}

#[test]
pub fn test_slotmap_can_allocate() {
    let _: LocklessSlotmap<usize, parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(128);
}

#[test]
pub fn test_slotmap_can_insert_without_realloc() {
    let slotmap: LocklessSlotmap<usize, parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(16);
    let tickets = (0..16).map(|i| slotmap.insert(i)).collect::<Vec<_>>();

    assert_eq!(slotmap.capacity(), 16);
    assert_eq!(slotmap.len(), 16);
    assert_eq!(slotmap.generation_limit_reached(), 0);

    for (i, &ticket) in tickets.iter().enumerate() {
        assert!(slotmap.get(ticket).is_some());
        assert_eq!(*slotmap.get(ticket).unwrap(), i);
    }
}

#[test]
pub fn test_slotmap_can_insert_with_realloc() {
    let slotmap: LocklessSlotmap<usize, parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(16);
    let tickets = (0..32).map(|i| slotmap.insert(i)).collect::<Vec<_>>();

    assert_eq!(slotmap.len(), 32);
    assert_eq!(slotmap.generation_limit_reached(), 0);
    assert!(slotmap.capacity() > 16);

    for (i, &ticket) in tickets.iter().enumerate() {
        assert!(slotmap.get(ticket).is_some());
        assert_eq!(*slotmap.get(ticket).unwrap(), i);
    }
}

#[test]
pub fn test_slotmap_does_not_leak() {
    let count = Arc::new(AtomicUsize::new(0));

    let slotmap: LocklessSlotmap<LeakDetector, parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(16);
    let _ = (0..28).map(|_| slotmap.insert(LeakDetector::new(count.clone()))).collect::<Vec<_>>();

    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 28);
    assert_eq!(slotmap.len(), 28);
    assert_eq!(slotmap.generation_limit_reached(), 0);
    assert!(slotmap.capacity() > 16);

    // Drop the slotmap and all the inserted elements
    drop(slotmap);

    // The slotmap and all the inserted elements should have been dropped
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
pub fn test_slotmap_erase() {
    let elem_count = Arc::new(AtomicUsize::new(0));

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(TEST_SEED);

    for _ in 0..TEST_ITERATIONS {
        let capacity = rng.gen_range(1..=256);
        let slotmap: LocklessSlotmap<(LeakDetector, u32), parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(capacity);

        // We choose randomly between inserting and erasing elements
        // so that the capacity is never exceeded.
        let mut inserted = Vec::new();
        let mut deleted_tickets = Vec::new();

        for _ in 0..TEST_ITERATIONS_SMALL {
            let mut should_insert = rng.gen_bool(0.5);
            if inserted.is_empty() {
                should_insert = true;
            }
            else if inserted.len() >= capacity {
                should_insert = false;
            }

            if should_insert {
                let element_value = rng.next_u32();
                let element = (LeakDetector::new(elem_count.clone()), element_value);
                let ticket = slotmap.insert(element);

                inserted.push((ticket, element_value));
            }
            else {
                // Remove a random element from the inserted elements
                let index = rng.gen_range(0..inserted.len());
                let (ticket, element_value) = inserted.remove(index);
                deleted_tickets.push(ticket);

                // Check that the element is still in the slotmap
                let element_ref = slotmap.get(ticket).unwrap();
                assert_eq!(element_ref.1, element_value);
                drop(element_ref); // If not dropped, the slotmap will be locked for this element

                // Erase the element from the slotmap
                let element = slotmap.erase(ticket).unwrap();
                assert_eq!(element.1, element_value);
                drop(element.0); // We drop the leak detector here to decrement the count

                // Check that the element is no longer in the slotmap
                assert_eq!(slotmap.len(), inserted.len());
                assert!(slotmap.get(ticket).is_none());
                assert_eq!(elem_count.load(std::sync::atomic::Ordering::SeqCst), inserted.len());
            }

            // Check that all the inserted elements are still in the slotmap
            for (ticket, element_value) in &inserted {
                let element_ref = slotmap.get(*ticket).unwrap();
                assert_eq!(element_ref.1, *element_value);
                drop(element_ref); // If not dropped, the slotmap will be locked for this element
            }

            // Check that all the deleted elements are no longer in the slotmap
            for ticket in &deleted_tickets {
                assert!(slotmap.get(*ticket).is_none());
            }
        }
    }
}

#[test]
pub fn test_slotmap_with_realloc() {
    let elem_count = Arc::new(AtomicUsize::new(0));

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(TEST_SEED);

    for _ in 0..TEST_ITERATIONS_TINY {
        let p: f64 = rng.gen_range(0.1..0.9);
        let slotmap: LocklessSlotmap<(LeakDetector, u32), parking_lot::RawRwLock> = crate::LocklessSlotmap::with_capacity(64);

        // We choose randomly between inserting and erasing elements
        // so that the capacity is never exceeded.
        let mut inserted = Vec::new();
        let mut deleted_tickets = Vec::new();

        for _ in 0..TEST_ITERATIONS_LARGE {
            let mut should_insert = rng.gen_bool(p);
            if inserted.is_empty() {
                should_insert = true;
            }

            if should_insert {
                let element_value = rng.next_u32();
                let element = (LeakDetector::new(elem_count.clone()), element_value);
                let ticket = slotmap.insert(element);

                inserted.push((ticket, element_value));
            }
            else {
                // Remove a random element from the inserted elements
                let index = rng.gen_range(0..inserted.len());
                let (ticket, element_value) = inserted.remove(index);
                deleted_tickets.push(ticket);

                // Check that the element is still in the slotmap
                let element_ref = slotmap.get(ticket).unwrap();
                assert_eq!(element_ref.1, element_value);
                drop(element_ref); // If not dropped, the slotmap will be locked for this element

                // Erase the element from the slotmap
                let element = slotmap.erase(ticket).unwrap();
                assert_eq!(element.1, element_value);
                drop(element.0); // We drop the leak detector here to decrement the count

                // Check that the element is no longer in the slotmap
                assert_eq!(slotmap.len(), inserted.len());
                assert!(slotmap.get(ticket).is_none());
                assert_eq!(elem_count.load(std::sync::atomic::Ordering::SeqCst), inserted.len());
            }

            // Check that all the inserted elements are still in the slotmap
            for (ticket, element_value) in &inserted {
                let element_ref = slotmap.get(*ticket).unwrap();
                assert_eq!(element_ref.1, *element_value);
                drop(element_ref); // If not dropped, the slotmap will be locked for this element
            }

            // Check that all the deleted elements are no longer in the slotmap
            for ticket in &deleted_tickets {
                assert!(slotmap.get(*ticket).is_none());
            }
        }
    }
}

#[test]
pub fn test_slotmap_cannot_erase_while_locked() {
    let slotmap: Arc<LocklessSlotmap<usize, RawRwLock>> = Arc::new(LocklessSlotmap::with_capacity(16));
    let ticket = slotmap.insert(0);
    let barrier_start = Arc::new(std::sync::Barrier::new(2));
    let barrier_end = Arc::new(std::sync::Barrier::new(2));

    {
        let slotmap = slotmap.clone();
        let barrier_start = barrier_start.clone();
        let barrier_end = barrier_end.clone();


        std::thread::spawn(move || {
            barrier_start.wait();

            let element_ref = slotmap.get(ticket).unwrap();
            assert_eq!(*element_ref, 0);

            barrier_end.wait();

            // Wait for at least one second to ensure that the slotmap is locked
            std::thread::sleep(std::time::Duration::from_secs(2));

            // Drop the element reference to unlock the slotmap
            drop(element_ref);
        });
    }

    barrier_start.wait();
    barrier_end.wait();

    // Determine the time it takes to erase the element
    let start = std::time::Instant::now();

    // Try to erase the element while the slotmap is locked
    let _ = slotmap.erase(ticket);

    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() >= 1, "The slotmap was not locked for at least one second");
}

fn test_slotmap_multithreaded_impl(with_erase: bool) {
    let elem_count = Arc::new(AtomicUsize::new(0));
    let next_element = Arc::new(AtomicUsize::new(0));
    let barrier_start = Arc::new(std::sync::Barrier::new(THREAD_COUNT));
    let mut root_rng = rand_chacha::ChaCha20Rng::seed_from_u64(TEST_SEED + 1);

    for _ in 0..2 {
        let slotmap: Arc<LocklessSlotmap<(LeakDetector, usize), parking_lot::RawRwLock>> = Arc::new(LocklessSlotmap::with_capacity(64));
        elem_count.store(0, std::sync::atomic::Ordering::Relaxed);
        next_element.store(0, std::sync::atomic::Ordering::Relaxed);

        let mut threads = Vec::with_capacity(THREAD_COUNT);
        for _ in 0..THREAD_COUNT {
            let slotmap = slotmap.clone();
            let elem_count = elem_count.clone();
            let next_element = next_element.clone();
            let barrier_start = barrier_start.clone();
            let p: f64 = root_rng.gen_range(0.1..0.9);
            let seed = root_rng.next_u64();

            let fn_callback = move || {
                barrier_start.wait();
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
                
                let mut inserted = Vec::new();
                let mut deleted_tickets = Vec::new();

                for _ in 0..TEST_ITERATIONS_VERY_LARGE {
                    let mut should_insert = if with_erase {
                        rng.gen_bool(p)
                    } else {
                        true
                    };
                    if inserted.is_empty() {
                        should_insert = true;
                    }

                    if should_insert {
                        let element_value = next_element.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let element = (LeakDetector::new(elem_count.clone()), element_value);
                        let ticket = slotmap.insert(element);

                        inserted.push((ticket, element_value));
                    }
                    else {
                        // Remove a random element from the inserted elements
                        let index = rng.gen_range(0..inserted.len());
                        let (ticket, element_value) = inserted.remove(index);

                        // Check that the element is still in the slotmap
                        let element_ref = slotmap.get(ticket).unwrap();
                        assert_eq!(element_ref.1, element_value);
                        drop(element_ref); // If not dropped, the slotmap will be locked for this element

                        // Erase the element from the slotmap
                        let element = slotmap.erase(ticket).unwrap();

                        // Check that the element is no longer in the slotmap
                        assert_eq!(element.1, element_value);
                        deleted_tickets.push(ticket);
                    }

                    // Perform test at intervals
                    if rng.gen_bool(0.1) {
                        // Check that all the inserted elements are still in the slotmap
                        for (ticket, element_value) in &inserted {
                            let element_ref = slotmap.get(*ticket).unwrap();
                            assert_eq!(element_ref.1, *element_value);
                            drop(element_ref); // If not dropped, the slotmap will be locked for this element
                        }

                        // Check that all the deleted elements are no longer in the slotmap
                        for ticket in &deleted_tickets {
                            assert!(slotmap.get(*ticket).is_none());
                        }
                    }

                    // At even rarer intervals, sleep the thread for a while
                    if rng.gen_bool(0.001) {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                }
            };

            threads.push(std::thread::spawn(fn_callback));
        }

        // Spawn the printing thread
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let logging_thread = {
            let stop = stop.clone();
            let slotmap = slotmap.clone();
            let next_element = next_element.clone();

            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    println!("Slotmap len: {}, capacity: {}, generation limit: {}, next element: {}", slotmap.len(), slotmap.capacity(), slotmap.generation_limit_reached(), next_element.load(std::sync::atomic::Ordering::Relaxed));
                }
            })
        };

        for thread in threads {
            thread.join().unwrap();
        }

        
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        logging_thread.join().unwrap();
        println!("Restarting test");
        if !with_erase {
            assert_eq!(slotmap.len(), THREAD_COUNT * TEST_ITERATIONS_VERY_LARGE);
            assert_eq!(next_element.load(std::sync::atomic::Ordering::Relaxed), THREAD_COUNT * TEST_ITERATIONS_VERY_LARGE);
        }
        drop(slotmap);

        assert_eq!(elem_count.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}

#[test]
pub fn test_slotmap_multithreaded_with_erase() {
    test_slotmap_multithreaded_impl(true);
}

// #[test]
// pub fn test_slotmap_multithreaded_without_erase() {
//     test_slotmap_multithreaded_impl(false);
// }
