use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

///
/// Describe the state a single slot can be in. It can
/// be either free, occupied or reserved.
///
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum State {
    Free {
        next_free_slot: Option<u32>,
        next_generation: u16,
    },
    Occupied {
        generation: u16,
    },
    Reserved,
}

impl Into<u64> for State {
    fn into(self) -> u64 {
        match self {
            State::Free { next_generation, next_free_slot } => {
                if next_free_slot.is_some() {
                    0x0000_0000_0000_0000 |
                        ((next_generation as u64) << 32) |
                        next_free_slot.unwrap() as u64
                }
                else {
                    0x1000_0000_0000_0000 |
                        ((next_generation as u64) << 32)
                }
            }
            State::Occupied { generation } => {
                0x2000_0000_0000_0000 | 
                    (generation as u64)
            }
            State::Reserved => {
                0x7fff_ffff_ffff_ffff
            }
        }
    }
}

impl From<u64> for State {
    fn from(state: u64) -> Self {
        match state & 0xffff_0000_0000_0000 {
            0x0000_0000_0000_0000 => {
                State::Free {
                    next_generation: (state >> 32) as u16,
                    next_free_slot: Some((state & 0xffff_ffff) as u32),
                }
            }
            0x1000_0000_0000_0000 => {
                State::Free {
                    next_generation: (state >> 32) as u16,
                    next_free_slot: None,
                }
            }
            0x2000_0000_0000_0000 => {
                State::Occupied {
                    generation: (state & 0xffff) as u16,
                }
            }
            _ => {
                State::Reserved
            }
        }
    }
}

///
/// An atomic state for a lockless slotmap.
/// 
/// This describe the state a single slot can be in. It can
/// be either free, occupied or reserved. 
/// Reserved is a special state that is used to indicate concurrent
/// access to the slot from another thread. This should prevent
/// other threads from accessing the slot until the reservation is
/// released.
///
pub(crate) struct AtomicState(AtomicU64);


impl AtomicState {
    pub fn new(state: State) -> Self {
        Self(AtomicU64::new(state.into()))
    }

    pub fn load(&self, order: Ordering) -> State {
        self.0.load(order).into()
    }

    pub fn store(&self, state: State, order: Ordering) {
        self.0.store(state.into(), order);
    }

    pub fn compare_exchange(
        &self,
        current: State,
        new: State,
        success: Ordering,
        failure: Ordering,
    ) -> Result<State, State> {
        match self.0.compare_exchange(current.into(), new.into(), success, failure) {
            Ok(new) => Ok(new.into()),
            Err(old) => Err(old.into()),
        }
    }
}

impl From<State> for AtomicState {
    fn from(state: State) -> Self {
        Self::new(state)
    }
}

///
/// NonZero atomic is a simple u32 atomic that can be converted to Option<NonZeroU32>
/// 
pub(crate) struct AtomicOptionU64(AtomicU64);

impl AtomicOptionU64 {
    fn convert_from(value: Option<u64>) -> u64 {
        assert!(value.unwrap_or(0) != u64::MAX, "Maximum value for NonOptionAtomicU64 is exceeded");
        value.map_or(u64::MAX, |value: u64| value as u64)
    }

    fn convert_to(value: u64) -> Option<u64> {
        if value == u64::MAX {
            None
        } else {
            Some(value)
        }
    }

    pub fn new(value: Option<u64>) -> Self {
        Self(AtomicU64::new(Self::convert_from(value)))
    }

    pub fn load(&self, order: Ordering) -> Option<u64> {
        Self::convert_to(self.0.load(order))
    }

    pub fn store(&self, value: Option<u64>, order: Ordering) {
        self.0.store(Self::convert_from(value), order);
    }

    pub fn compare_exchange(
        &self,
        current: Option<u64>,
        new: Option<u64>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<u64>, Option<u64>> {
        let current = Self::convert_from(current);
        let new = Self::convert_from(new);

        match self.0.compare_exchange(current, new, success, failure) {
            Ok(new) => Ok(Self::convert_to(new)),
            Err(old) => Err(Self::convert_to(old)),
        }
    }
}

///
/// NonZero atomic is a simple u32 atomic that can be converted to Option<NonZeroU32>
/// 
pub(crate) struct AtomicOptionU32(AtomicU32);

impl AtomicOptionU32 {
    fn convert_from(value: Option<u32>) -> u32 {
        assert!(value.unwrap_or(0) != u32::MAX, "Maximum value for NonOptionAtomicu32 is exceeded");
        value.map_or(u32::MAX, |value: u32| value as u32)
    }

    fn convert_to(value: u32) -> Option<u32> {
        if value == u32::MAX {
            None
        } else {
            Some(value)
        }
    }

    pub fn new(value: Option<u32>) -> Self {
        Self(AtomicU32::new(Self::convert_from(value)))
    }

    pub fn load(&self, order: Ordering) -> Option<u32> {
        Self::convert_to(self.0.load(order))
    }

    pub fn store(&self, value: Option<u32>, order: Ordering) {
        self.0.store(Self::convert_from(value), order);
    }

    pub fn compare_exchange(
        &self,
        current: Option<u32>,
        new: Option<u32>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<u32>, Option<u32>> {
        let current = Self::convert_from(current);
        let new = Self::convert_from(new);

        match self.0.compare_exchange(current, new, success, failure) {
            Ok(new) => Ok(Self::convert_to(new)),
            Err(old) => Err(Self::convert_to(old)),
        }
    }
}