use std::{cell::UnsafeCell, mem::MaybeUninit, sync::atomic::AtomicUsize};

use utils::{AtomicOptionU32, AtomicOptionU64, AtomicState, State};

#[cfg(test)]
pub mod tests;
pub mod utils;

///
/// This constant represents the maximum number of elements
/// per block at the maximum size of the slotmap.
/// 
const MAX_ELEMENTS_PER_BLOCK: usize = 32768;
const _: () = assert!(MAX_ELEMENTS_PER_BLOCK < std::u32::MAX as usize, "The MAX_ELEMENTS_PER_BLOCK must be less than u32::MAX (constraint due to AtomicState)");

///
/// Structure that represents a ticket for a slot in the slotmap.
/// 
/// The ticket can then be used to access the slot in the slotmap. 
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotmapTicket(u64);

impl SlotmapTicket {
    pub(crate) fn new(block_index: u16, slot_index: u32, generation: u16) -> Self
    {
        assert!(block_index <= std::u16::MAX, "The block index has exceeded the maximum value of u16");
        assert!(generation <= std::u16::MAX, "The generation has exceeded the maximum value of u16");

        Self(
            block_index as u64 |
            ((generation as u64) << 16) |
            ((slot_index as u64) << 32)
        )
    }

    pub(crate) fn block_index(&self) -> u16
    {
        (self.0 & 0xffff) as u16
    }

    pub(crate) fn generation(&self) -> u16
    {
        ((self.0 & 0xffff_0000) >> 16) as u16
    }

    pub(crate) fn slot_index(&self) -> u32
    {
        ((self.0 & 0xffff_ffff_0000_0000) >> 32) as u32
    }
}

///
/// Structure that represents an element currently stored in the slotmap. 
/// 
/// The structure provides a reference to the element, and ensures that the
/// element cannot be removed from the slotmap while the reference is alive. Any
/// call to the slotmap that would remove the element will block until the reference
/// is dropped.
///
pub struct SlotmapEntry<'a, T> {
    atomic_ref: &'a AtomicUsize,
    data: &'a T,
}

impl <'a, T> SlotmapEntry<'a, T> {
    ///
    /// Gets a reference to the element stored in the slotmap.
    /// 
    /// # Returns
    /// A reference to the element stored in the slotmap.
    ///
    pub fn get<'b: 'a>(&'b self) -> &'b T
    {
        self.data
    }
}

impl <'a, T> Drop for SlotmapEntry<'a, T> {
    fn drop(&mut self)
    {
        self.atomic_ref.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl <'a, T> std::ops::Deref for SlotmapEntry<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target
    {
        self.get()
    }
}


struct LocklessStateEntry
{
    refcount: AtomicUsize,
    state: AtomicState,
}

struct LocklessSlotmapBlock<T>
{
    elements: Box<[UnsafeCell<MaybeUninit<T>>]>,
    states: Box<[LocklessStateEntry]>,
    next_free_slot: AtomicOptionU32,
    next_non_saturated_block: AtomicOptionU64,
}

impl <T> LocklessSlotmapBlock<T> {
    fn new(size: usize) -> Self
    {
        debug_assert!(size <= MAX_ELEMENTS_PER_BLOCK, "The size of the block must be less than or equal to MAX_ELEMENTS_PER_BLOCK");
        debug_assert!(size < std::u32::MAX as usize, "The size of the block must be less than u32::MAX (constraint due to AtomicState)");
        debug_assert!(size > 0, "The size of the block must be greater than 0");

        let mut elements = Vec::with_capacity(size);
        let mut states = Vec::with_capacity(size);

        for i in 0..size {
            elements.push(UnsafeCell::new(MaybeUninit::uninit()));
            states.push(LocklessStateEntry {
                refcount: AtomicUsize::new(0),
                state: State::Free {
                    next_generation: 0,
                    next_free_slot: if i < size - 1 {
                        Some(i as u32 + 1)
                    } else {
                        None
                    }
                }.into(),
            });
        }

        Self {
            elements: elements.into_boxed_slice(),
            states: states.into_boxed_slice(),
            next_free_slot: AtomicOptionU32::new(Some(0)),
            next_non_saturated_block: AtomicOptionU64::new(None),
        }
    }
}


pub struct LocklessSlotmap<T, R>
where
    T: Sized + Send + Sync,
    R: lock_api::RawRwLock,
{
    blocks: lock_api::RwLock<R, Vec<LocklessSlotmapBlock<T>>>,
    next_non_saturated_block: AtomicOptionU64,
    next_block_size: AtomicUsize,
    capacity: AtomicUsize,
    len: AtomicUsize,
    generation_limit_reached: AtomicUsize,
}

impl <T, R> LocklessSlotmap<T, R>
where
    T: Sized + Send + Sync,
    R: lock_api::RawRwLock,
{
    fn grow(current_size: usize) -> usize
    {
        std::cmp::min(MAX_ELEMENTS_PER_BLOCK, current_size + (current_size >> 1))
    }

    fn alloc_block(&self)
    {
        // First we need to acquire the write lock (exclusive access)
        // to the blocks.
        // NOTE: This will lock the entire slotmap, so it is not
        //       recommended. Already retrieved references won't
        //       be affected by this lock.
        let mut blocks = self.blocks.write();

        // Check the current next_non_saturated_block
        let next_non_saturated_block = self.next_non_saturated_block.load(std::sync::atomic::Ordering::SeqCst);
        if next_non_saturated_block.is_some() {
            return;
        }

        // Determine the size of the next block
        let next_block_size = self.next_block_size.load(std::sync::atomic::Ordering::Relaxed);
        self.next_block_size.store(Self::grow(next_block_size), std::sync::atomic::Ordering::Relaxed);

        // Allocate the elements and states for the new block
        let new_block = LocklessSlotmapBlock::new(next_block_size);
        new_block.next_non_saturated_block.store(None, std::sync::atomic::Ordering::Relaxed);

        // Add the block to the blocks
        let block_index = blocks.len();
        assert!(block_index < std::u16::MAX as usize, "The number of blocks has exceeded the maximum value of u16");
        blocks.push(new_block);

        if self.next_non_saturated_block.compare_exchange(
            None,
            Some(block_index as u64),
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::Relaxed
        ).is_err() {
            blocks.pop();
            return;
        }

        // Increment the capacity and size
        self.capacity.fetch_add(next_block_size, std::sync::atomic::Ordering::Relaxed);
    }

    ///
    /// Creates a new slotmap with the default capacity (64 elements per block). Preallocates the first block.
    /// 
    /// # Returns
    /// A new slotmap with the default capacity.
    /// 
    pub fn new() -> Self
    {
        Self::with_capacity(64)
    }

    ///
    /// Creates a new slotmap with the specified capacity. Preallocates the first block.
    /// 
    /// # Arguments
    /// * `capacity` - The capacity of the slotmap.
    /// 
    /// # Returns
    /// A new slotmap with the specified capacity.
    /// 
    /// # Panics
    /// Panics if the capacity is greater than MAX_ELEMENTS_PER_BLOCK or if the capacity is 0.
    /// 
    pub fn with_capacity(capacity: usize) -> Self
    {
        assert!(capacity <= MAX_ELEMENTS_PER_BLOCK, "The capacity of the slotmap must be less than or equal to MAX_ELEMENTS_PER_BLOCK");
        assert!(capacity > 0, "The capacity of the slotmap must be greater than 0");

        let object = Self {
            blocks: lock_api::RwLock::new(Vec::new()),
            next_non_saturated_block: AtomicOptionU64::new(None),
            next_block_size: AtomicUsize::new(capacity),
            capacity: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            generation_limit_reached: AtomicUsize::new(0),
        };
        object.alloc_block(); // Pre-allocate the first block
        object
    }

    pub fn insert(&self, value: T) -> SlotmapTicket
    {
        let backoff = crossbeam::utils::Backoff::new();
        loop {
            // Attempt to find a free slot in the slotmap
            let block_index = self.next_non_saturated_block.load(std::sync::atomic::Ordering::SeqCst);

            // If there is no block available, allocate a new block
            let block_index = if let Some(block_index) = block_index {
                usize::try_from(block_index).unwrap()
            }
            else {
                self.alloc_block();
                continue; // Retry
            };

            // Acquire the read lock (shared access) to the blocks
            let blocks = self.blocks.read();
            let block = &blocks[block_index as usize];

            // Attempt to find a free slot in the block
            let slot_index = block.next_free_slot.load(std::sync::atomic::Ordering::SeqCst);
            let slot_index = if let Some(slot_index) = slot_index {
                slot_index
            }
            else {
                // Another thread has made a progress, retry
                // FIXME: Race-Condition: test_multithreaded_insertion_and_removal stuck here
                // because the next_block is always saturated
                backoff.spin();
                continue;
            };

            let slot_state = &block.states[slot_index as usize];

            // Attempt to acquire the slot at slot_index
            let state = slot_state.state.load(std::sync::atomic::Ordering::SeqCst);
            let (next_generation, next_free_slot) = match state {
                State::Free { next_generation, next_free_slot } => {
                    (next_generation, next_free_slot)
                },
                _ => {
                    // Another thread has made a progress, retry
                    backoff.spin();
                    continue;
                }
            };

            // We then need to attempt to transition the state of the slot to Reserved
            if slot_state.state.compare_exchange(
                state,
                State::Reserved,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst
            ).is_err() {
                // Another thread has made a progress, retry
                backoff.spin();
                continue;
            }

            // We then update the next_free_slot of the block
            if block.next_free_slot.compare_exchange(
                Some(slot_index),
                next_free_slot,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst
            ).is_err() {
                // Another thread has made a progress, retry
                backoff.spin();
                slot_state.state.store(state, std::sync::atomic::Ordering::SeqCst);
                continue;
            }

            // If there is no next_free_slot, we need to update the next_non_saturated_block
            if next_free_slot.is_none() {
                if self.next_non_saturated_block.compare_exchange(
                    Some(block_index as u64),
                    block.next_non_saturated_block.load(std::sync::atomic::Ordering::SeqCst),
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst
                ).is_err() {
                    // Another thread has made a progress, retry
                    backoff.spin();
                    slot_state.state.store(state, std::sync::atomic::Ordering::SeqCst);
                    block.next_free_slot.store(Some(slot_index), std::sync::atomic::Ordering::SeqCst);
                    continue;
                }
            }

            // We then need to initialize the element at slot_index
            unsafe {
                let element = block.elements[slot_index as usize].get().as_mut().unwrap();
                element.write(value);
            }

            // We then need to transition the state of the slot to Occupied
            if slot_state.state.compare_exchange(
                State::Reserved,
                State::Occupied { generation: next_generation },
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst
            ).is_err() {
                panic!("Race condition detected, this is a bug, please report it.");
            }
            
            // Finally create the ticket
            self.len.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return SlotmapTicket::new(block_index as u16, slot_index, next_generation);
        }
    }

    ///
    /// Gets a reference to the element stored in the slotmap at the specified ticket.
    /// 
    /// # Arguments
    /// * `ticket` - The ticket of the element in the slotmap.
    /// 
    /// # Returns
    /// A reference to the element stored in the slotmap at the specified ticket.
    /// 
    pub fn get(&self, ticket: SlotmapTicket) -> Option<SlotmapEntry<'_, T>> {
        let block_index = ticket.block_index();
        let slot_index = ticket.slot_index();
        let ticket_generation = ticket.generation();

        // Acquire the read lock (shared access) to the blocks
        let blocks = self.blocks.read();
        let block = &blocks[usize::from(block_index)];

        // Acquire the state of the slot
        let slot_state = &block.states[slot_index as usize];
        let state = slot_state.state.load(std::sync::atomic::Ordering::SeqCst);

        // Check if the slot is occupied and the generation matches
        match state {
            State::Occupied { generation } if generation == ticket_generation => (),
            _ => return None,
        }

        // Increment the refcount of the slot
        slot_state.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Recheck the state of the slot
        let new_state = slot_state.state.load(std::sync::atomic::Ordering::SeqCst);
        if new_state != state {
            // Another thread has made a progress, decrement the refcount
            slot_state.refcount.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            return None;
        }

        // Get the reference to the element

        // SAFETY: The slot is occupied and the generation matches, we have incremented the refcount
        //         and rechecked the state of the slot **AFTERWARDS**, therefore the reference must be valid.
        let element = unsafe {
            block.elements[slot_index as usize].get().as_ref().unwrap().assume_init_ref()
        };

        // SAFETY: Boxed are never removed unless the LocklessSlotmap is dropped
        //         therefore the reference lifetime can be extended to the lifetime of the LocklessSlotmap
        let refcount: &'_ AtomicUsize = unsafe {
            std::mem::transmute(&slot_state.refcount)
        };

        // Return the slotmap entry
        Some(SlotmapEntry {
            atomic_ref: refcount,
            data: element,
        })
    }

    ///
    /// Remove an element from the slotmap at the specified ticket.
    /// 
    /// # Arguments
    /// * `ticket` - The ticket of the element in the slotmap.
    /// 
    /// # Returns
    /// The element stored in the slotmap at the specified ticket.
    /// 
    pub fn erase(&self, ticket: SlotmapTicket) -> Option<T> {
        let block_index = ticket.block_index();
        let slot_index = ticket.slot_index();
        let ticket_generation = ticket.generation();

        // Acquire the read lock (shared access) to the blocks
        let blocks = self.blocks.read();

        // Acquire the state of the slot
        let block = &blocks[usize::from(block_index)];

        // Acquire the state of the slot
        let slot_state = &block.states[slot_index as usize];

        // Begin of the critical section
        let backoff = crossbeam::utils::Backoff::new();
        'critical: loop {
            // Check that the slot is occupied and the generation matches
            let state = slot_state.state.load(std::sync::atomic::Ordering::SeqCst);
            match state {
                State::Occupied { generation } if generation == ticket_generation => (),
                _ => break 'critical None, // The slot is not occupied or the generation does not match
            }

            // Attempt to transition the state of the slot to Reserved
            if slot_state.state.compare_exchange(
                state,
                State::Reserved,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst
            ).is_err() {
                // Another thread has made a progress, retry
                backoff.spin();
                continue;
            }

            // Second loop: Await for the refcount to hit 0
            'zeroref: loop {
                let refcount = slot_state.refcount.load(std::sync::atomic::Ordering::SeqCst);
                if refcount == 0 {
                    break 'zeroref;
                }

                // Another thread has made a progress, retry
                backoff.snooze();
            }

            // Retrieve the element
            let element = unsafe {
                block.elements[slot_index as usize].get().as_mut().unwrap().assume_init_read()
            };

            // We then attempt to transition the state of the slot to Free
            if let Some(next_generation) = ticket_generation.checked_add(1) {
                // We update the next_free_slot of the block so that it points to this slot (currently reserved)
                let next_free_slot = 'update_slot: loop {
                    let next_free_slot = block.next_free_slot.load(std::sync::atomic::Ordering::SeqCst);

                    if block.next_free_slot.compare_exchange(
                        next_free_slot,
                        Some(slot_index),
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst
                    ).is_err() {
                        // Another thread has made a progress, retry
                        backoff.spin();
                        continue;
                    }

                    // If next_free_slot is None, that means that this blog was dangling
                    // and now it is not anymore, we need to update the next_non_saturated_block
                    if next_free_slot.is_none() {
                        let next_non_saturated_block = 'update_block: loop {
                            let next_non_saturated_block = self.next_non_saturated_block.load(std::sync::atomic::Ordering::SeqCst);
                            
                            // Attempt to update the next_non_saturated_block
                            if self.next_non_saturated_block.compare_exchange(
                                next_non_saturated_block,
                                Some(block_index as u64),
                                std::sync::atomic::Ordering::SeqCst,
                                std::sync::atomic::Ordering::SeqCst
                            ).is_err() {
                                // Another thread has made a progress, retry
                                backoff.spin();
                                continue;
                            }

                            // We finally update the next_free_slot of the slot
                            break 'update_block next_non_saturated_block;
                        };

                        // We then update the next_non_saturated_block of the block
                        block.next_non_saturated_block.store(next_non_saturated_block, std::sync::atomic::Ordering::SeqCst);
                    };

                    // We finally update the next_free_slot of the slot
                    break 'update_slot next_free_slot;
                };

                // The generation has not reached the maximum value, we can reuse this slot
                if slot_state.state.compare_exchange(
                    State::Reserved,
                    State::Free {
                        next_generation: next_generation,
                        next_free_slot: next_free_slot,
                    },
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst
                ).is_err() {
                    panic!("refcount is 0, the slot is reserved, no overwriting should occur, this is a bug, please report it.");
                }
            }
            else {
                // The generation has reached the maximum value, we won't be reusing
                // this slot, therefore we leave it as Reserved
                self.generation_limit_reached.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            self.len.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            break 'critical Some(element);
        }
    }

    ///
    /// Get the capacity of the slotmap.
    /// 
    pub fn capacity(&self) -> usize
    {
        self.capacity.load(std::sync::atomic::Ordering::SeqCst)
    }

    ///
    /// Get the number of elements in the slotmap.
    /// 
    pub fn len(&self) -> usize
    {
        self.len.load(std::sync::atomic::Ordering::SeqCst)
    }

    ///
    /// Number of slots that have reached the maximum generation. Those slot
    /// won't be available for reuse. 
    ///
    pub fn generation_limit_reached(&self) -> usize
    {
        self.generation_limit_reached.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl <T, R> Drop for LocklessSlotmap<T, R>
where
    T: Sized + Send + Sync,
    R: lock_api::RawRwLock,
{
    fn drop(&mut self)
    {
        // Acquire the write lock (exclusive access) to the blocks
        let blocks = self.blocks.write();

        // Drop all the blocks
        for block in blocks.iter() {
            for (slot_state, slot_data) in block.states.iter().zip(block.elements.iter()) {
                // First we need to acquire the state of the slot
                let state = slot_state.state.load(std::sync::atomic::Ordering::SeqCst);

                // If the slot is occupied, drop the element
                match state {
                    State::Reserved => {
                        // State can only be Reserved if they reached the maximum
                        // generation, therefore the refcount must be 0
                        let refcount = slot_state.refcount.load(std::sync::atomic::Ordering::SeqCst);
                        assert_eq!(refcount, 0, "The refcount of the slot is not 0, this is a bug, please report it.");
                    }
                    State::Free { .. } => (),
                    State::Occupied { .. } => {
                        // Check that the refcount is 0
                        let refcount = slot_state.refcount.load(std::sync::atomic::Ordering::SeqCst);
                        assert_eq!(refcount, 0, "The refcount of the slot is not 0, this is a bug, please report it.");

                        // Drop the element
                        // SAFETY: The slot is occupied, the refcount is 0, the slot is being dropped
                        //         therefore the element can be safely dropped.
                        unsafe {
                            slot_data.get().as_mut().unwrap().assume_init_drop();
                        }
                    }
                }
            }
        }
    }
}

unsafe impl <T, R> Send for LocklessSlotmap<T, R>
where
    T: Sized + Send + Sync,
    R: lock_api::RawRwLock,
{}

unsafe impl <T, R> Sync for LocklessSlotmap<T, R>
where
    T: Sized + Send + Sync,
    R: lock_api::RawRwLock,
{}
