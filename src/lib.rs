//! # Lockless Slotmap
//! 
//! Lockless Slotmap is a lockless implementation of a slotmap. A slotmap is a data structure that
//! allows for stable references to elements while providing fast insertion (in O(1) time) and
//! removal (in O(1) time).
//! 
//! This implementation is (mostly) lockless, meaning that it can be used in a high performance
//! environment where locks are not desired. The only place where locks are used is when the slotmap
//! becomes saturated and a new block needs to be allocated. Because of the ever-growing exponential
//! size of the blocks, this should be a rare occurrence.
//! 
//! # Limitations
//! 
//! Each slot in the slotmap can only be reused so many times. 16-bit generation numbers are kept as
//! guard against ABA problems, this means that each slot can only be reused 65536 times (after which
//! the slot is considered "dead" and will not be reused which can lead to slowly increasing memory
//! usage). Therefore for very long running applications with high insertion and removal rates, this
//! implementation may not be suitable.
//! 
//! # Example
//! 
//! ```
//! use rust_lockless_slotmap::LocklessSlotmap;
//! use std::sync::Arc;
//! use parking_lot::RawRwLock;
//! 
//! let slotmap: Arc<LocklessSlotmap<usize, RawRwLock>> = Arc::new(LocklessSlotmap::new());
//! let ticket = slotmap.insert(42);
//! 
//! let slotmap_clone = Arc::clone(&slotmap);
//! let handle = std::thread::spawn(move || {
//!    let entry = slotmap_clone.get(ticket).unwrap();
//!    assert_eq!(*entry, 42);
//!    drop(entry); // Deadlock if this is not dropped
//! 
//!    slotmap_clone.insert(45);
//!    slotmap_clone.erase(ticket);
//! });
//! 
//! handle.join().unwrap();
//! 
//! assert_eq!(slotmap.len(), 1);
//! assert!(slotmap.get(ticket).is_none());
//! ```
//! 

use std::{cell::UnsafeCell, mem::MaybeUninit, sync::atomic::AtomicUsize};

use utils::{AtomicOptionU32, AtomicOptionU64, AtomicState, State};

#[cfg(test)]
mod tests;
mod utils;

/// Maximum number of elements per allocation block.
/// 
/// [`SlotmapTicket`] allows for stable references to elements in the slotmap, as such
/// dynamic resizing of the slotmap is not possible. In order to achieve this, the
/// slotmap is divided into blocks, each block having a fixed number of elements. 
/// 
/// The slotmap can only grow by adding new blocks. The number of element per block
/// starts at 64 (default) and can grow up to a limit defined by [`MAX_ELEMENTS_PER_BLOCK`].
/// 
/// The maximum theoretical number of elements per block is 2^32 (due to constraints on
/// the [`SlotmapTicket`] structure).
pub const MAX_ELEMENTS_PER_BLOCK: usize = 32768;
const _: () = assert!(MAX_ELEMENTS_PER_BLOCK < std::u32::MAX as usize, "The MAX_ELEMENTS_PER_BLOCK must be less than u32::MAX (constraint due to AtomicState)");

/// Holds a ticket (think of it as a reference) to an element stored in the slotmap.
/// 
/// The ticket is used to access the element stored in the slotmap. The ticket is
/// created when the element is inserted into the slotmap and is used to access
/// the element until the element is removed from the slotmap.
/// 
/// Notice that the slotmap implementation ensures that each ticket is unique. If
/// an element is removed from the slotmap, the ticket is invalidated and cannot
/// be used, even if the slot is reused.
/// 
/// You can retrieve the element corresponding to the ticket using the
/// [`LocklessSlotmap::get`] which will also guarantee that the element is not
/// removed while it is being accessed. This ticket makes no such guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotmapTicket {
    block_index: u16,
    generation: u16,
    slot_index: u32,
}

impl SlotmapTicket {
    pub(crate) fn new(block_index: u16, slot_index: u32, generation: u16) -> Self
    {
        assert!(block_index <= std::u16::MAX, "The block index has exceeded the maximum value of u16");
        assert!(generation <= std::u16::MAX, "The generation has exceeded the maximum value of u16");

        Self {
            block_index: block_index,
            generation: generation,
            slot_index: slot_index,
        }
    }

    pub(crate) fn block_index(&self) -> u16
    {
        self.block_index
    }

    pub(crate) fn generation(&self) -> u16
    {
        self.generation
    }

    pub(crate) fn slot_index(&self) -> u32
    {
        self.slot_index
    }
}

/// SlotmapEntry is a reference to an element stored in the slotmap.
/// 
/// The SlotmapEntry is created when the element is accessed in the slotmap. It
/// ensures that the element cannot be removed while there is a thread actively
/// accessing it.
/// 
/// # Note
/// 
/// It is the responsibility of the user to ensure that the SlotmapEntry is dropped
/// prior to removing the element from the slotmap. Failure to do so will result in
/// a deadlock, where the erasing method will wait indefinitely for the 
/// SlotmapEntry to be dropped.
/// 
/// # Example
/// ```
/// use rust_lockless_slotmap::LocklessSlotmap;
/// use parking_lot::RawRwLock;
/// 
/// let slotmap: LocklessSlotmap<usize, RawRwLock> = LocklessSlotmap::new();
/// let ticket = slotmap.insert(42);
/// 
/// {
///    let entry = slotmap.get(ticket).unwrap();
///    assert_eq!(*entry, 42);
/// }
/// ```
/// 
pub struct SlotmapEntry<'a, T> {
    atomic_ref: &'a AtomicUsize,
    data: &'a T,
}

impl <'a, T> SlotmapEntry<'a, T> {
    /// Get a reference to the element stored in the slotmap.
    /// 
    /// This reference cannot outlive the protection of the SlotmapEntry.
    /// Therefore all access to this element are guaranteed to be safe.
    /// 
    /// # Returns
    /// A reference to the element stored in the slotmap.
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

/// LocklessSlotmap is a lockless implementation of a slotmap.
/// 
/// A slotmap is a data structure that allows for stable references to elements
/// while providing fast insertion (in O(1) time) and removal (in O(1) time).
/// 
/// This implementation is (mostly) lockless, meaning that it can be used in a
/// high performance environment where locks are not desired. The only place where
/// locks are used is when the slotmap becomes saturated and a new block needs to
/// be allocated. Because of the ever-growing exponential size of the blocks, this
/// should be a rare occurrence.
/// 
/// # Limitations
/// 
/// Each slot in the slotmap can only be reused so many times. 16-bit generation
/// numbers are kept as guard against ABA problems, this means that each slot can
/// only be reused 65536 times (after which the slot is considered "dead" and will
/// not be reused which can lead to slowly increasing memory usage). Therefore for
/// very long running applications with high insertion and removal rates, this
/// implementation may not be suitable.
/// 
/// # Implementation
/// 
/// Internally, the slotmap is divided into blocks, each block containing a fixed
/// number of elements. When the slotmap is saturated, a new block is allocated
/// without invalidating all the already existing blocks. This allows for fast
/// insertion and removal of elements.
/// 
/// Blocks grow exponentially in size, starting at 64 elements (default) and
/// growing up to a maximum of [`MAX_ELEMENTS_PER_BLOCK`] elements.
/// 
/// # Note
/// 
/// In the current implementation, the insertion of new elements takes the place of
/// the most recently removed element. At high loads this behavior can lead to
/// excessive memory fragmentation. This behavior may be changed in the future.
/// 
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

    /// Creates a new slotmap with the default capacity of 64 elements.
    /// 
    /// # Returns
    /// A new slotmap with the default capacity of 64 elements.
    /// 
    /// # Panics
    /// Panics if the allocation of the first block fails.
    pub fn new() -> Self
    {
        Self::with_capacity(64)
    }

    /// Creates a new slotmap with the specified capacity.
    /// 
    /// # Arguments
    /// * `capacity` - The capacity (number of elements) of the slotmap. This capacity is limited
    ///                to [`MAX_ELEMENTS_PER_BLOCK`] elements. However, this limit should allow for
    ///                a slotmap with a capacity of up to 2^32 elements.
    /// 
    /// # Returns
    /// A new slotmap with the specified capacity.
    /// 
    /// # Panics
    /// Panics if the capacity is greater than [`MAX_ELEMENTS_PER_BLOCK`].
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

    /// Inserts a new element into the slotmap.
    /// 
    /// Atomically inserts a new element into the slotmap. The element is stored in the slotmap
    /// and a ticket is returned that can be used to access the element.
    /// 
    /// # Arguments
    /// * `value` - The value to insert into the slotmap.
    /// 
    /// # Returns
    /// A ticket that can be used to access the element in the slotmap. The value of the ticket
    /// can then be accessed using the [`LocklessSlotmap::get`] method.
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

    /// Get an element from the slotmap at the specified ticket.
    /// 
    /// Retrieves the element stored in the slotmap at the specified ticket. The ticket
    /// is invalidated after the element is removed from the slotmap. For more information
    /// you can refer to the [`SlotmapEntry`] structure.
    /// 
    /// # Arguments
    /// * `ticket` - The ticket of the element in the slotmap. Tickets are invalidated
    ///              after the element is removed from the slotmap. See [`SlotmapTicket`]
    ///              for more details.
    /// 
    /// # Returns
    /// An [`Option<T>`] containing a [`SlotmapEntry`] to the element stored in the slotmap
    /// at the specified ticket. If the ticket is invalid or the element has been removed
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

    /// Erase an element from the slotmap at the specified ticket.
    /// 
    /// Removes the element stored in the slotmap at the specified ticket. The ticket is
    /// invalidated after the element is removed from the slotmap. For more information
    /// you can refer to the [`SlotmapEntry`] structure.
    /// 
    /// # Deadlocks
    /// 
    /// This method will wait for all [`SlotmapEntry`] corresponding to the ticket to be
    /// dropped before removing the element from the slotmap. Special care should be taken
    /// to ensure that the thread calling this method is not holding a [`SlotmapEntry`]
    /// corresponding to the ticket or a deadlock will occur.
    /// 
    /// # Arguments
    /// * `ticket` - The ticket of the element in the slotmap. Tickets are invalidated
    /// 
    /// # Returns
    /// An [`Option<T>`] containing the element stored in the slotmap at the specified ticket or
    /// [`Option::None`] if the ticket is invalid or the element has already been removed.
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

    /// Get the maximum number of elements that can be stored in the slotmap.
    /// 
    /// # Returns
    /// The maximum number of elements that can be stored in the slotmap. 
    pub fn capacity(&self) -> usize
    {
        self.capacity.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the number of elements stored in the slotmap.
    /// 
    /// Notice that in a multithreaded environment, the number of elements stored in the slotmap
    /// can change between the time this method is called and the time the result is used, therefore
    /// this method should be used as an approximation.
    /// 
    /// # Returns
    /// The number of elements stored in the slotmap. 
    pub fn len(&self) -> usize
    {
        self.len.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Number of times the generation limit has been reached.
    /// 
    /// As discussed in the limitations of the [`LocklessSlotmap`] structure, each slot can only
    /// be reused so many times (2^16 times). When the generation limit is reached, the slot is
    /// considered "dead" and will not be reused. This method returns the number of times dead
    /// slots have been encountered.
    /// 
    /// Notice that in a multithreaded environment, the number of times the generation limit has
    /// been reached can change between the time this method is called and the time the result is
    /// used, therefore this method should be used as an approximation.
    /// 
    /// # Returns
    /// The number of times the generation limit has been reached.
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
