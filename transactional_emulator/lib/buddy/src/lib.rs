use core::fmt::{self, Debug, Formatter};
use std::collections::BTreeSet;

/// Power-of-two sizes.
///
/// The maximum size representable is 2 ** 64.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Size(u32);

impl Size {
    pub const fn from_log2(log2: u32) -> Self {
        assert!(log2 <= 64);
        Self(log2)
    }

    pub const fn in_log2(self) -> u32 {
        self.0
    }

    pub const fn in_bytes(self) -> u64 {
        1 << self.0
    }

    pub const fn next_size(self) -> Size {
        Self(self.0 + 1)
    }

    pub const fn from_bytes(size: u64) -> Size {
        Self(size.next_power_of_two().ilog2())
    }
}

/// An abstract region of memory.
#[derive(Clone, Copy)]
pub struct MemoryRange {
    base: u64,
    size: u64,
}

impl MemoryRange {
    /// Construct a memory range.
    ///
    /// Returns `None` if the limit (base + size) overflows the address space.
    pub fn checked_new(base: u64, size: u64) -> Option<Self> {
        base.checked_add(size)?;

        Some(MemoryRange { base, size })
    }

    pub fn new(base: u64, size: u64) -> Self {
        Self::checked_new(base, size).unwrap()
    }

    pub const fn base(&self) -> u64 {
        self.base
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub fn base_sub(self, size: u64) -> Option<Self> {
        Some(Self {
            base: self.base.checked_sub(size)?,
            size: self.size.checked_add(size)?,
        })
    }

    pub fn base_add(self, size: u64) -> Option<Self> {
        Some(Self {
            base: self.base.checked_add(size)?,
            size: self.size.checked_sub(size)?,
        })
    }

    pub fn limit_sub(self, size: u64) -> Option<Self> {
        Some(Self {
            base: self.base,
            size: self.size.checked_sub(size)?,
        })
    }

    pub fn limit_add(self, size: u64) -> Option<Self> {
        Some(Self {
            base: self.base,
            size: self.size.checked_add(size)?,
        })
    }

    fn base_align_down(self, align: Size) -> Option<Self> {
        let misalignment = self.base & (align.in_bytes() - 1);
        self.base_sub(misalignment)
    }

    fn base_align_up(self, align: Size) -> Option<Self> {
        let misalignment = self.base.wrapping_neg() & (align.in_bytes() - 1);
        self.base_add(misalignment)
    }

    fn limit_align_down(self, align: Size) -> Option<Self> {
        let misalignment = self.base.wrapping_add(self.size) & (align.in_bytes() - 1);
        self.limit_sub(misalignment)
    }

    fn limit_align_up(self, align: Size) -> Option<Self> {
        let misalignment =
            self.base.wrapping_add(self.size).wrapping_neg() & (align.in_bytes() - 1);
        self.limit_add(misalignment)
    }

    fn align_expand(self, align: Size) -> Option<Self> {
        self.base_align_down(align)?.limit_align_up(align)
    }

    fn align_shrink(self, align: Size) -> Option<Self> {
        self.base_align_up(align)?.limit_align_down(align)
    }
}

/// A bitmap backed by `[u8]` slice.
#[repr(transparent)]
struct Bitmap([u8]);

impl Bitmap {
    /// Get size in bytes needed for at least given number of entries.
    pub const fn size_needed(entries: usize) -> usize {
        entries.div_ceil(8)
    }

    /// Allocate a bitmap on heap.
    pub fn new(entries: usize) -> Box<Self> {
        let size = Self::size_needed(entries);
        let slice = vec![0u8; size].into_boxed_slice();
        // SAFETY: `Bitmap` is a `#[repr(transparent)]` wrapper of `[u8]`.
        unsafe { core::mem::transmute(slice) }
    }

    /// Test if a bit is set.
    pub fn test(&self, index: usize) -> bool {
        let byte = &self.0[index / 8];
        let mask = 1 << (index % 8);
        *byte & mask != 0
    }

    /// Flip a bit and return the original.
    pub fn flip(&mut self, index: usize) -> bool {
        let byte = &mut self.0[index / 8];
        let mask = 1 << (index % 8);
        *byte ^= mask;
        *byte & mask == 0
    }
}

pub struct BuddyAllocator {
    #[allow(dead_code)]
    range: MemoryRange,
    min_block: Size,
    free_blocks: Vec<BTreeSet<u64>>,
    metadata: Vec<(Box<Bitmap>, u64)>,
}

impl Debug for BuddyAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_map();
        for (i, free_block) in self.free_blocks.iter().enumerate() {
            debug.entry(&(i as u32 + self.min_block.in_log2()), free_block);
        }
        Ok(())
    }
}

impl BuddyAllocator {
    fn max_block(&self) -> Size {
        Size::from_log2(self.min_block.in_log2() + self.metadata.len() as u32)
    }

    // Test if a bit is set.
    fn metadata_test(&self, size: Size, ptr: u64) -> bool {
        let (ref bitmap, base) =
            self.metadata[(size.in_log2() - self.min_block.in_log2()) as usize];
        let bitmap_index = ptr.checked_sub(base).unwrap() >> size.next_size().in_log2();
        bitmap.test(bitmap_index.try_into().unwrap())
    }

    // Flip a bit and return the original.
    fn metadata_flip(&mut self, size: Size, offset: u64) -> bool {
        let (ref mut bitmap, base) =
            self.metadata[(size.in_log2() - self.min_block.in_log2()) as usize];
        let bitmap_index = offset.checked_sub(base).unwrap() >> size.next_size().in_log2();
        bitmap.flip(bitmap_index.try_into().unwrap())
    }

    /// Allocate a block of exact size.
    fn allocate_exact(&mut self, size: Size) -> Option<u64> {
        let block =
            self.free_blocks[(size.in_log2() - self.min_block.in_log2()) as usize].pop_first()?;
        self.metadata_flip(size, block);
        Some(block)
    }

    /// Deallocate a block of exact size.
    ///
    /// If the buddy is also free, then both blocks are merged and the parent block is returned to be deallocated.
    fn deallocate_exact(&mut self, size: Size, block: u64) -> Option<u64> {
        // If the bit is original set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        if self.metadata_flip(size, block) {
            self.free_blocks[(size.in_log2() - self.min_block.in_log2()) as usize]
                .remove(&(block ^ size.in_bytes()));
            return Some(block & !size.in_bytes());
        }

        self.free_blocks[(size.in_log2() - self.min_block.in_log2()) as usize].insert(block);
        None
    }

    /// Try to find if buddy is allocated for an already-allocated `block`.
    fn is_buddy_allocated(&mut self, size: Size, block: u64) -> bool {
        // If the bit is set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        !self.metadata_test(size, block)
    }

    #[cold]
    fn allocate_max(&mut self, size: Size) -> Option<u64> {
        // Too large to allocate
        if size != self.max_block() {
            return None;
        }

        self.free_blocks.last_mut().unwrap().pop_first()
    }

    #[cold]
    fn deallocate_max(&mut self, size: Size, mut ptr: u64) {
        let max_block = self.max_block();
        for _ in 0..(1 << (size.in_log2() - max_block.in_log2())) {
            self.free_blocks.last_mut().unwrap().insert(ptr);
            ptr += max_block.in_bytes();
        }
    }

    pub fn allocate(&mut self, size: Size) -> Option<u64> {
        if size >= self.max_block() {
            return self.allocate_max(size);
        }

        let size = std::cmp::max(size, self.min_block);
        match self.allocate_exact(size) {
            Some(v) => Some(v),
            None => {
                let ptr = self.allocate(Size::from_log2(size.in_log2() + 1))?;
                self.deallocate_exact(size, ptr + size.in_bytes())
                    .map(|_| unreachable!());
                Some(ptr)
            }
        }
    }

    pub fn deallocate(&mut self, size: Size, ptr: u64) {
        let max_block = Size::from_log2(self.min_block.in_log2() + self.metadata.len() as u32);
        if size >= max_block {
            return self.deallocate_max(size, ptr);
        }

        let size = std::cmp::max(size, self.min_block);
        match self.deallocate_exact(size, ptr) {
            None => (),
            Some(v) => {
                self.deallocate(Size::from_log2(size.in_log2() + 1), v);
            }
        }
    }

    pub fn shrink(&mut self, size: Size, mut new_size: Size, block: u64) {
        while new_size < size {
            self.deallocate(new_size, block + new_size.in_bytes());
            new_size = Size::from_log2(new_size.in_log2() + 1);
        }
    }

    /// Check if `block` can grow in place.
    pub fn can_grow(&mut self, size: Size, new_size: Size, block: u64) -> bool {
        // Check for alignment, if unaligned then this is definitely not possible
        let aligned_ptr = block as u64 >> new_size.in_log2() << new_size.in_log2();
        if block as u64 != aligned_ptr {
            return false;
        }

        let mut test_size = size;
        while test_size < new_size {
            if self.is_buddy_allocated(test_size, block) {
                return false;
            }
            test_size = Size::from_log2(test_size.in_log2() + 1);
        }

        true
    }

    /// Commit the block growth.
    pub fn grow(&mut self, mut size: Size, new_size: Size, block: u64) {
        debug_assert!(self.can_grow(size, new_size, block));

        while size < new_size {
            let v = self.deallocate_exact(size, block);
            assert_eq!(v, Some(block));
            size = Size::from_log2(size.in_log2() + 1);
        }
    }

    /// Add memory to this allocator.
    pub fn add_memory(&mut self, memory: MemoryRange) -> Option<()> {
        let mut memory = memory.base_align_up(self.min_block)?;

        while memory.size != 0 {
            // Get the next naturally aligned block contained within the memory.
            let max_addr_align = Size::from_log2(std::cmp::min(
                memory.base.trailing_zeros(),
                63 - memory.size.leading_zeros(),
            ));
            self.deallocate(max_addr_align, memory.base);
            memory = memory.base_add(max_addr_align.in_bytes())?;
        }
        Some(())
    }

    /// Create a new buddy allocator.
    ///
    /// The minimum block size (i.e. allocation size) is specified, and the maximum operable range is also
    /// specified. Necessary structure tracking the allocation would be allocated on heap.
    pub fn new(min_block: Size, range: MemoryRange) -> Option<Self> {
        // Ensrue that the range is aligned to the minimum block size.
        let range = range.align_shrink(min_block)?;

        // Compute the maximum possible block size for the given range.
        let max_block = std::cmp::max(
            Size::from_log2((range.base ^ (range.base.wrapping_add(range.size))).trailing_zeros()),
            min_block,
        );
        let mut metadata = Vec::with_capacity((max_block.in_log2() - min_block.in_log2()) as usize);

        for i in min_block.in_log2()..max_block.in_log2() {
            // Allocate bitmap for tracking this size.
            // We only need 1 bit for each buddy pair.
            let range_for_bitmap = range.align_expand(Size::from_log2(i + 1))?;
            let bitmap = Bitmap::new((range_for_bitmap.size >> (i + 1)).try_into().ok()?);
            metadata.push((bitmap, range_for_bitmap.base()));
        }

        Some(BuddyAllocator {
            range,
            min_block,
            free_blocks: vec![
                BTreeSet::new();
                (max_block.in_log2() - min_block.in_log2() + 1) as usize
            ],
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// 64-byte minimum block (a common minimum allocation granularity).
    fn k64() -> Size {
        Size::from_log2(6)
    }

    /// A buddy allocator over `[base, base + size)` with `min`-sized blocks,
    /// already seeded with the whole range as free memory.
    fn fresh(min: Size, base: u64, size: u64) -> BuddyAllocator {
        let mut a = BuddyAllocator::new(min, MemoryRange::new(base, size)).unwrap();
        a.add_memory(MemoryRange::new(base, size)).unwrap();
        a
    }

    // ---------- Size ----------

    #[test]
    fn test_size_in_bytes() {
        assert_eq!(Size::from_log2(0).in_bytes(), 1);
        assert_eq!(Size::from_log2(6).in_bytes(), 64);
        assert_eq!(Size::from_log2(10).in_bytes(), 1024);
    }

    #[test]
    fn test_size_log2_roundtrip() {
        for l in 0..=63 {
            assert_eq!(Size::from_log2(l).in_log2(), l);
        }
    }

    #[test]
    fn test_size_from_bytes_rounds_up_to_pow2() {
        assert_eq!(Size::from_bytes(1).in_log2(), 0);
        assert_eq!(Size::from_bytes(63).in_log2(), 6); // next_power_of_two(63) == 64
        assert_eq!(Size::from_bytes(64).in_log2(), 6);
        assert_eq!(Size::from_bytes(65).in_log2(), 7); // next_power_of_two(65) == 128
    }

    #[test]
    fn test_size_next_size() {
        assert_eq!(Size::from_log2(6).next_size(), Size::from_log2(7));
    }

    // ---------- MemoryRange ----------

    #[test]
    fn test_range_checked_new_overflow() {
        assert!(MemoryRange::checked_new(0, u64::MAX).is_some());
        assert!(MemoryRange::checked_new(1, u64::MAX).is_none());
        assert!(MemoryRange::checked_new(u64::MAX, 1).is_none());
    }

    #[test]
    fn test_range_accessors_and_shifts() {
        let r = MemoryRange::new(64, 256);
        assert_eq!(r.base(), 64);
        assert_eq!(r.size(), 256);

        let a = r.base_add(64).unwrap();
        assert_eq!((a.base(), a.size()), (128, 192));

        let s = r.base_sub(64).unwrap();
        assert_eq!((s.base(), s.size()), (0, 320));
        assert!(r.base_sub(128).is_none()); // base would underflow below 0

        let la = r.limit_add(10).unwrap();
        assert_eq!((la.base(), la.size()), (64, 266));

        let ls = r.limit_sub(56).unwrap();
        assert_eq!((ls.base(), ls.size()), (64, 200));
    }

    // ---------- BuddyAllocator: allocate / split ----------

    #[test]
    fn test_alloc_drains_in_increasing_address_order() {
        let mut a = fresh(k64(), 0, 1024);
        // The first 64-byte allocation splits the 1024 block down; addresses
        // then drain in increasing order until the region is exhausted.
        assert_eq!(a.allocate(k64()), Some(0));
        assert_eq!(a.allocate(k64()), Some(64));
        assert_eq!(a.allocate(k64()), Some(128));
        assert_eq!(a.allocate(k64()), Some(192));

        let mut count = 4;
        let mut prev = 192;
        while let Some(p) = a.allocate(k64()) {
            assert!(p % 64 == 0 && p < 1024);
            assert!(p > prev);
            prev = p;
            count += 1;
        }
        assert_eq!(count, 16); // 1024 / 64
        assert_eq!(a.allocate(k64()), None);
    }

    // ---------- BuddyAllocator: deallocate / coalesce ----------

    #[test]
    fn test_dealloc_coalesces_buddies() {
        let mut a = fresh(k64(), 0, 128); // exactly two 64-byte blocks
        let p0 = a.allocate(k64()).unwrap();
        let p1 = a.allocate(k64()).unwrap();
        assert_eq!((p0, p1), (0, 64));
        assert_eq!(a.allocate(k64()), None);

        // Freeing both buddies merges them back into a 128-byte block.
        a.deallocate(k64(), p0);
        a.deallocate(k64(), p1);
        assert_eq!(a.allocate(Size::from_log2(7)), Some(0));
    }

    #[test]
    fn test_no_coalesce_while_buddy_busy() {
        let mut a = fresh(k64(), 0, 128);
        let p0 = a.allocate(k64()).unwrap();
        let _p1 = a.allocate(k64()).unwrap();

        a.deallocate(k64(), p0); // buddy still allocated -> no merge
        assert_eq!(a.allocate(Size::from_log2(7)), None); // cannot form a 128
        assert_eq!(a.allocate(k64()), Some(0)); // but the freed 64 is reusable
    }

    // ---------- BuddyAllocator: shrink ----------

    #[test]
    fn test_shrink_frees_the_tail() {
        let mut a = fresh(k64(), 0, 256);
        let p = a.allocate(Size::from_log2(8)).unwrap(); // the whole 256 region
        assert_eq!(p, 0);
        assert_eq!(a.allocate(k64()), None);

        a.shrink(Size::from_log2(8), k64(), p); // 256 -> 64, frees [64, 256)
        assert_eq!(a.allocate(k64()), Some(64));
        assert_eq!(a.allocate(k64()), Some(128));
        assert_eq!(a.allocate(k64()), Some(192));
        assert_eq!(a.allocate(k64()), None);
    }

    // ---------- BuddyAllocator: can_grow / grow ----------

    #[test]
    fn test_can_grow_then_grow_in_place() {
        let mut a = fresh(k64(), 0, 128);
        let p = a.allocate(k64()).unwrap(); // 64@0; buddy 64@64 stays free
        assert_eq!(p, 0);

        assert!(a.can_grow(k64(), Size::from_log2(7), p));
        a.grow(k64(), Size::from_log2(7), p); // now owns the whole 128
        assert_eq!(a.allocate(k64()), None);
    }

    #[test]
    fn test_cannot_grow_when_buddy_busy() {
        let mut a = fresh(k64(), 0, 128);
        let p = a.allocate(k64()).unwrap();
        let _q = a.allocate(k64()).unwrap(); // buddy busy
        assert!(!a.can_grow(k64(), Size::from_log2(7), p));
    }

    // ---------- proptest: structural invariants ----------

    proptest! {
        #[test]
        fn prop_live_allocations_never_overlap(ops in prop::collection::vec(0u8..3, 0..200)) {
            const CAP: u64 = 4096;
            let mut a = fresh(k64(), 0, CAP);
            let mut live: Vec<(u64, u64)> = Vec::new(); // (ptr, size_in_bytes)

            for op in ops {
                match op {
                    0 | 1 => {
                        let size = if op == 0 { k64() } else { Size::from_log2(7) };
                        if let Some(p) = a.allocate(size) {
                            let bytes = size.in_bytes();
                            prop_assert_eq!(p % bytes, 0); // aligned to its own size
                            prop_assert!(p + bytes <= CAP); // within the region
                            for &(q, qb) in &live {
                                // disjoint from every other live allocation
                                prop_assert!(p + bytes <= q || q + qb <= p);
                            }
                            live.push((p, bytes));
                        }
                    }
                    _ => {
                        if let Some((p, bytes)) = live.pop() {
                            a.deallocate(Size::from_bytes(bytes), p);
                        }
                    }
                }
            }
        }
    }
}
