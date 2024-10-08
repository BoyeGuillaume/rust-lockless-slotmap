# rust-lockless-slotmap

## Lockless Slotmap

Lockless Slotmap is a lockless implementation of a slotmap. A slotmap is a data structure that
allows for stable references to elements while providing fast insertion (in O(1) time) and
removal (in O(1) time).

This implementation is (mostly) lockless, meaning that it can be used in a high performance
environment where locks are not desired. The only place where locks are used is when the slotmap
becomes saturated and a new block needs to be allocated. Because of the ever-growing exponential
size of the blocks, this should be a rare occurrence.

## Limitations

Each slot in the slotmap can only be reused so many times. 16-bit generation numbers are kept as
guard against ABA problems, this means that each slot can only be reused 65536 times (after which
the slot is considered "dead" and will not be reused which can lead to slowly increasing memory
usage). Therefore for very long running applications with high insertion and removal rates, this
implementation may not be suitable.

## Example

```rust
use rust_lockless_slotmap::LocklessSlotmap;
use std::sync::Arc;
use parking_lot::RawRwLock;

let slotmap: Arc<LocklessSlotmap<usize, RawRwLock>> = Arc::new(LocklessSlotmap::new());
let ticket = slotmap.insert(42);

let slotmap_clone = Arc::clone(&slotmap);
let handle = std::thread::spawn(move || {
   let entry = slotmap_clone.get(ticket).unwrap();
   assert_eq!(*entry, 42);
   drop(entry); // Deadlock if this is not dropped

   slotmap_clone.insert(45);
   slotmap_clone.erase(ticket);
});

handle.join().unwrap();

assert_eq!(slotmap.len(), 1);
assert!(slotmap.get(ticket).is_none());
```


License: MIT
