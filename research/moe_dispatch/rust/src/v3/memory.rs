//! Finite byte pool and 1RW 16-byte banks. Addresses are physical, not fluid rates.
#[derive(Clone)]
pub struct Pool {
    pub capacity: usize,
    pub ranges: Vec<(usize, usize)>,
    pub used: usize,
    pub peak: usize,
    pub banks: usize,
    pub writes: u64,
    pub reads: u64,
    pub conflicts: u64,
}
impl Pool {
    pub fn new(bytes: usize, banks: usize) -> Self {
        Self {
            capacity: bytes,
            ranges: vec![(0, bytes)],
            used: 0,
            peak: 0,
            banks,
            writes: 0,
            reads: 0,
            conflicts: 0,
        }
    }
    pub fn alloc(&mut self, n: usize) -> Option<usize> {
        let i = self.ranges.iter().position(|&(_, b)| b >= n)?;
        let (a, b) = self.ranges[i];
        if b == n {
            self.ranges.remove(i);
        } else {
            self.ranges[i] = (a + n, b - n);
        }
        self.used += n;
        self.peak = self.peak.max(self.used);
        Some(a)
    }
    pub fn alloc_partition(&mut self,n:usize,lo:usize,hi:usize)->Option<usize>{
        let (i,start)=self.ranges.iter().enumerate().find_map(|(i,&(a,b))|{let s=a.max(lo);if s.checked_add(n)?<= (a+b).min(hi){Some((i,s))}else{None}})?;
        let (a,b)=self.ranges.remove(i);if start>a{self.ranges.push((a,start-a));}if start+n<a+b{self.ranges.push((start+n,a+b-start-n));}self.ranges.sort();self.used+=n;self.peak=self.peak.max(self.used);Some(start)
    }
    pub fn release(&mut self, a: usize, n: usize) {
        self.used -= n;
        self.ranges.push((a, n));
        self.ranges.sort();
        let mut merged: Vec<(usize, usize)> = vec![];
        for &(p, b) in &self.ranges {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == p {
                    last.1 += b;
                    continue;
                }
            }
            merged.push((p, b));
        }
        self.ranges = merged;
    }
    pub fn word(&mut self, addr: usize, busy: &mut [bool], write: bool) -> bool {
        let bank = (addr / 16) % self.banks;
        if busy[bank] {
            self.conflicts += 1;
            return false;
        }
        busy[bank] = true;
        if write {
            self.writes += 16;
        } else {
            self.reads += 16;
        }
        true
    }
}
pub struct BankPort {
    pub banks: usize,
    pub used: Vec<bool>,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub conflicts: u64,
}
impl BankPort {
    pub fn new(n: usize) -> Self {
        Self {
            banks: n,
            used: vec![false; n],
            read_bytes: 0,
            write_bytes: 0,
            conflicts: 0,
        }
    }
    pub fn clear(&mut self) {
        self.used.fill(false);
    }
    pub fn transfer_spans(
        &mut self,
        spans: &[(usize, usize)],
        done: &mut usize,
        max_bytes: usize,
        write: bool,
    ) -> usize {
        let start = *done;
        let total: usize = spans.iter().map(|s| s.1).sum();
        for _ in 0..max_bytes / 16 {
            if *done >= total {
                break;
            }
            let mut offset = *done;
            let mut addr = None;
            for &(base, len) in spans {
                if offset < len {
                    addr = Some(base + offset);
                    break;
                }
                offset -= len;
            }
            let bank = (addr.expect("gather offset in valid spans") / 16) % self.banks;
            if self.used[bank] {
                self.conflicts += 1;
                break;
            }
            self.used[bank] = true;
            *done = (*done + 16).min(total);
            if write {
                self.write_bytes += 16;
            } else {
                self.read_bytes += 16;
            }
        }
        *done - start
    }
    pub fn transfer(
        &mut self,
        base: usize,
        done: &mut usize,
        bytes: usize,
        max_bytes: usize,
        write: bool,
    ) -> usize {
        let start = *done;
        for _ in 0..max_bytes / 16 {
            if *done >= bytes {
                break;
            }
            let bank = ((base + *done) / 16) % self.banks;
            if self.used[bank] {
                self.conflicts += 1;
                break;
            }
            self.used[bank] = true;
            *done = (*done + 16).min(bytes);
            if write {
                self.write_bytes += 16;
            } else {
                self.read_bytes += 16;
            }
        }
        *done - start
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_pool() {
        let mut p = Pool::new(128, 8);
        assert_eq!(p.alloc(64), Some(0));
        assert_eq!(p.alloc(64), Some(64));
        assert_eq!(p.alloc(32), None);
        p.release(0, 64);
        p.release(64, 64);
        assert_eq!(p.ranges, vec![(0, 128)]);
    }
    #[test]
    fn shared_read_write() {
        let mut p = Pool::new(128, 8);
        let mut busy = vec![false; 8];
        assert!(p.word(0, &mut busy, true));
        assert!(!p.word(128, &mut busy, false));
        assert!(p.word(16, &mut busy, false));
    }
    #[test]
    fn private_slot_partitions_do_not_borrow_each_others_free_bytes(){
        let mut p=Pool::new(4*4096,8);let a=p.alloc_partition(2*4096,0,2*4096).unwrap();assert!(p.alloc_partition(4096,0,2*4096).is_none());let b=p.alloc_partition(4096,2*4096,4*4096).unwrap();assert_eq!(b,2*4096);p.release(a,2*4096);p.release(b,4096);assert_eq!(p.used,0);
    }
}
