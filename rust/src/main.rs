// Shopping Sessions (HCMUT TMT): Rust pipeline, std only.
// Implements rules.md. Reads $DATA_DIR, writes Parquet to $OUTPUT_DIR.
mod parquet;

use std::fs;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const DAY: i64 = 86_400_000;
const GAP: u32 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const PBITS: u32 = 8;
const NPART: usize = 1 << PBITS;
const PSHIFT: u32 = 64 - PBITS;
const CHUNK: usize = 4 << 20;

// ------------------------------------------------------------------ files
extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
}

fn map_file(path: &Path) -> &'static [u8] {
    let f = fs::File::open(path).expect("open");
    let len = f.metadata().expect("meta").len() as usize;
    if len == 0 {
        return &[];
    }
    // PROT_READ = 1, MAP_PRIVATE = 2
    let p = unsafe { mmap(std::ptr::null_mut(), len, 1, 2 | 0x8000, f.as_raw_fd(), 0) };
    if p as isize == -1 || p.is_null() {
        return Box::leak(fs::read(path).expect("read").into_boxed_slice());
    }
    unsafe { std::slice::from_raw_parts(p, len) }
}

fn map_all(paths: &[PathBuf], nt: usize) -> Vec<&'static [u8]> {
    let cells: Vec<SyncCell<&'static [u8]>> = paths.iter().map(|_| SyncCell::new(&[][..])).collect();
    // biggest first for balance
    let mut order: Vec<usize> = (0..paths.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(fs::metadata(&paths[i]).map(|m| m.len()).unwrap_or(0)));
    par_for(order.len(), nt, |_, j| {
        let i = order[j];
        unsafe { *cells[i].get() = map_file(&paths[i]) };
    });
    cells.into_iter().map(|c| c.into_inner()).collect()
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.ends_with(".csv"))
                        .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

#[inline]
fn find_nl(d: &[u8], mut i: usize) -> usize {
    let n = d.len();
    while i + 8 <= n {
        let v = u64::from_le_bytes(d[i..i + 8].try_into().unwrap());
        let x = v ^ 0x0a0a_0a0a_0a0a_0a0a;
        let t = x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
        if t != 0 {
            return i + (t.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n {
        if d[i] == b'\n' {
            return i;
        }
        i += 1;
    }
    n
}

/// First line start >= x; position 0 means "skip the header line".
fn adjust(d: &[u8], x: usize) -> usize {
    if x == 0 {
        (find_nl(d, 0) + 1).min(d.len())
    } else if x >= d.len() {
        d.len()
    } else {
        (find_nl(d, x - 1) + 1).min(d.len())
    }
}

// ------------------------------------------------------------------ CSV dialect
#[allow(dead_code)]
#[inline(always)]
fn zmask(x: u64) -> u64 {
    // exact: high bit set for each zero byte of x
    let y = (x & 0x7F7F_7F7F_7F7F_7F7F).wrapping_add(0x7F7F_7F7F_7F7F_7F7F);
    !(y | x | 0x7F7F_7F7F_7F7F_7F7F)
}

#[inline(always)]
fn masks64(d: &[u8], i: usize) -> (u64, u64, u64) {
    // d[i..i+64] must be in bounds
    #[cfg(target_feature = "avx2")]
    unsafe {
        use core::arch::x86_64::*;
        let p = d.as_ptr().add(i) as *const __m256i;
        let a = _mm256_loadu_si256(p);
        let b = _mm256_loadu_si256(p.add(1));
        let nl = _mm256_set1_epi8(b'\n' as i8);
        let cm = _mm256_set1_epi8(b',' as i8);
        let qt = _mm256_set1_epi8(b'"' as i8);
        let m = |x: __m256i, c: __m256i| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, c)) as u32 as u64 };
        (m(a, nl) | (m(b, nl) << 32), m(a, cm) | (m(b, cm) << 32), m(a, qt) | (m(b, qt) << 32))
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let (mut mn, mut mc, mut mq) = (0u64, 0u64, 0u64);
        for w in 0..8 {
            let v = u64::from_le_bytes(d[i + 8 * w..i + 8 * w + 8].try_into().unwrap());
            let pack = |z: u64| -> u64 {
                // high bit of each byte -> 8 bits
                ((z >> 7) & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56
            };
            mn |= pack(zmask(v ^ 0x0A0A_0A0A_0A0A_0A0A)) << (8 * w);
            mc |= pack(zmask(v ^ 0x2C2C_2C2C_2C2C_2C2C)) << (8 * w);
            mq |= pack(zmask(v ^ 0x2222_2222_2222_2222)) << (8 * w);
        }
        (mn, mc, mq)
    }
}

/// For every well-formed row (exactly K fields) of the lines starting in [s, e)
/// call f(fields). One pass over 64-byte blocks finds newlines, commas and quotes.
#[inline(always)]
fn for_rows<const K: usize>(d: &'static [u8], s: usize, e: usize, mut f: impl FnMut([&'static [u8]; K])) {
    let n = d.len();
    let p0 = adjust(d, s);
    let end = adjust(d, e);
    if p0 >= end {
        return;
    }
    let base = d.as_ptr();
    // field bounds; bad rows are flagged and skipped at the newline
    let mut fs = [0u32; K];
    let mut fe = [0u32; K];
    let mut k = 0usize;
    let mut st = p0;
    let mut q = 0u32;
    let mut bad = false;
    let mut tail: [u8; 64];
    let mut i = p0;
    // close field [st, j) with q quotes into slot k
    macro_rules! close {
        ($j:expr) => {{
            let j = $j;
            if k < K {
                let (a, z) = if q == 0 {
                    (st, j)
                } else if q == 2 && j - st >= 2 && unsafe { *base.add(st) == b'"' && *base.add(j - 1) == b'"' } {
                    (st + 1, j - 1)
                } else {
                    bad = true;
                    (st, st)
                };
                unsafe {
                    *fs.get_unchecked_mut(k) = (a - p0) as u32;
                    *fe.get_unchecked_mut(k) = (z - p0) as u32;
                }
            } else {
                bad = true;
            }
            k += 1;
        }};
    }
    loop {
        let (mn, mc, mq) = if i + 64 <= n {
            masks64(d, i)
        } else {
            let r = n - i;
            tail = [b'\n'; 64];
            tail[..r].copy_from_slice(&d[i..n]);
            masks64(&tail, 0)
        };
        let mut m = mn | mc | mq;
        while m != 0 {
            let b = m.trailing_zeros();
            let bit = 1u64 << b;
            let j = i + b as usize;
            if mq & bit != 0 {
                q += 1;
            } else {
                let nl = mn & bit != 0;
                let j2 = if nl { j.min(n) } else { j };
                close!(j2);
                st = j + 1;
                q = 0;
                if nl {
                    if !bad && k == K {
                        let mut out: [&'static [u8]; K] = [&d[0..0]; K];
                        for x in 0..K {
                            out[x] = unsafe {
                                std::slice::from_raw_parts(base.add(p0 + fs[x] as usize), (fe[x] - fs[x]) as usize)
                            };
                        }
                        f(out);
                    }
                    if st >= end {
                        return;
                    }
                    k = 0;
                    bad = false;
                }
            }
            m &= m - 1;
        }
        i += 64;
    }
}


// ------------------------------------------------------------------ hashing
#[inline]
fn fmix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

#[inline]
fn hash_seed(b: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ (b.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut i = 0;
    while i + 8 <= b.len() {
        let v = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        i += 8;
    }
    if i < b.len() {
        let mut t = [0u8; 8];
        t[..b.len() - i].copy_from_slice(&b[i..]);
        let v = u64::from_le_bytes(t);
        h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    fmix(h)
}

/// Two independent hashes in one pass (for event-id dedup identity).
#[inline]
fn hash2(b: &[u8]) -> (u64, u32) {
    let l = b.len() as u64;
    let mut h = 0x243F_6A88_85A3_08D3 ^ l.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut g = 0x1357_9BDF_2468_ACE0 ^ l.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    let mut i = 0;
    let mut step = |v: u64| {
        h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        g = (g ^ v.rotate_left(23)).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        g ^= g >> 31;
    };
    while i + 8 <= b.len() {
        step(u64::from_le_bytes(b[i..i + 8].try_into().unwrap()));
        i += 8;
    }
    if i < b.len() {
        let mut t = [0u8; 8];
        t[..b.len() - i].copy_from_slice(&b[i..]);
        step(u64::from_le_bytes(t));
    }
    let h = fmix(h);
    (if h == 0 { 1 } else { h }, (fmix(g ^ 0x5bd1_e995) >> 17) as u32)
}

#[inline]
fn hash(b: &[u8]) -> u64 {
    let h = hash_seed(b, 0x243F_6A88_85A3_08D3);
    if h == 0 {
        1
    } else {
        h
    }
}

/// Exact-key open-addressing table: bytes -> u32.
struct Table {
    mask: usize,
    keys: Vec<(u64, &'static [u8])>,
    vals: Vec<u32>,
}

impl Table {
    fn new(n: usize) -> Table {
        let cap = (n * 2 + 16).next_power_of_two();
        Table { mask: cap - 1, keys: vec![(0, &[][..]); cap], vals: vec![0; cap] }
    }
    #[inline]
    fn get_h(&self, h: u64, k: &[u8]) -> Option<u32> {
        let mut i = (h as usize) & self.mask;
        loop {
            let (kh, kk) = self.keys[i];
            if kh == 0 {
                return None;
            }
            if kh == h && kk == k {
                return Some(self.vals[i]);
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn get(&self, k: &[u8]) -> Option<u32> {
        self.get_h(hash(k), k)
    }
    /// Insert if absent; returns the stored value.
    fn insert(&mut self, k: &'static [u8], v: u32) -> u32 {
        let h = hash(k);
        let mut i = (h as usize) & self.mask;
        loop {
            let (kh, kk) = self.keys[i];
            if kh == 0 {
                self.keys[i] = (h, k);
                self.vals[i] = v;
                return v;
            }
            if kh == h && kk == k {
                return self.vals[i];
            }
            i = (i + 1) & self.mask;
        }
    }
}

// ------------------------------------------------------------------ timestamps
#[inline]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[inline]
fn dig(c: u8) -> Option<i64> {
    if c.is_ascii_digit() {
        Some((c - b'0') as i64)
    } else {
        None
    }
}

#[inline]
fn num(b: &[u8]) -> Option<i64> {
    let mut v = 0i64;
    for &c in b {
        v = v * 10 + dig(c)?;
    }
    Some(v)
}

#[inline(always)]
fn all_digits(x: u64) -> bool {
    let a = x & 0x7F7F_7F7F_7F7F_7F7F;
    let ge0 = a.wrapping_add(0x5050_5050_5050_5050);
    let ge10 = a.wrapping_add(0x4646_4646_4646_4646);
    (!x & ge0 & !ge10 & 0x8080_8080_8080_8080) == 0x8080_8080_8080_8080
}

/// Value of 8 ASCII digits (first byte most significant), little-endian load.
#[inline(always)]
fn eight(x: u64) -> u64 {
    let mut v = x.wrapping_sub(0x3030_3030_3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    (((v & 0x0000_00FF_0000_00FF).wrapping_mul(100 + (1_000_000u64 << 32)))
        .wrapping_add(((v >> 16) & 0x0000_00FF_0000_00FF).wrapping_mul(1 + (10_000u64 << 32))))
        >> 32
}

static MONTH_START: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
const MS_Y0: i64 = 1900;
const MS_NY: i64 = 300;

#[inline]
fn days_ym(y: i64, m: i64) -> i64 {
    if (MS_Y0..MS_Y0 + MS_NY).contains(&y) {
        let t = MONTH_START.get_or_init(|| {
            let mut v = Vec::with_capacity((MS_NY * 12) as usize);
            for yy in MS_Y0..MS_Y0 + MS_NY {
                for mm in 1..=12 {
                    v.push(days_from_civil(yy, mm, 1) as i32);
                }
            }
            v
        });
        t[((y - MS_Y0) * 12 + m - 1) as usize] as i64
    } else {
        days_from_civil(y, m, 1)
    }
}

/// Epoch ms, or None for a bad timestamp. Huge F3 values map to i64::MAX.
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    if !((n == 20 || n == 25) && b[4] == b'-') {
        if n <= 16 {
            // F3 fast path: right-align into 16 '0's, check and convert 8 digits at a time
            let mut buf = [b'0'; 16];
            buf[16 - n..].copy_from_slice(b);
            let hi = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let lo = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            if !all_digits(hi) || !all_digits(lo) {
                return None;
            }
            return Some((eight(hi) as i64) * 100_000_000 + eight(lo) as i64);
        }
        // F3: digits only
        let mut v: i64 = 0;
        let mut sig = 0;
        for &c in b {
            if !c.is_ascii_digit() {
                return None;
            }
            if sig > 0 || c != b'0' {
                sig += 1;
                if sig > 18 {
                    v = i64::MAX; // out of any window
                    continue;
                }
                v = v * 10 + (c - b'0') as i64;
            }
        }
        return Some(v);
    }
    // bytes 0..20: "DDDD-DD-DDTDD:DD:DD?" checked 8 bytes at a time
    let w0 = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let w2 = u32::from_le_bytes(b[16..20].try_into().unwrap()) as u64;
    if (w0 & 0xFF00_00FF_0000_0000) != 0x2D00_002D_0000_0000
        || (w1 & 0x0000_FF00_00FF_0000) != 0x0000_3A00_0054_0000
        || (w2 & 0xFF) != 0x3A
    {
        return None;
    }
    #[inline(always)]
    fn digits(x: u64, m: u64) -> bool {
        let a = x & 0x7F7F_7F7F_7F7F_7F7F;
        let ge0 = a.wrapping_add(0x5050_5050_5050_5050); // high bit: a >= '0'
        let ge10 = a.wrapping_add(0x4646_4646_4646_4646); // high bit: a >= ':'
        let ok = !x & ge0 & !ge10 & 0x8080_8080_8080_8080;
        ok & m == m
    }
    if !digits(w0, 0x0080_8000_8080_8080) || !digits(w1, 0x8080_0080_8000_8080) || !digits(w2, 0x0080_8000) {
        return None;
    }
    #[inline(always)]
    fn d2(x: u64, i: u32) -> i64 {
        ((x >> (8 * i)) & 0xFF) as i64 * 10 + ((x >> (8 * i + 8)) & 0xFF) as i64 - 528
    }
    let y = d2(w0, 0) * 100 + d2(w0, 2);
    let mo = d2(w0, 5);
    let d = d2(w1, 0);
    let h = d2(w1, 3);
    let mi = d2(w1, 6);
    let s = d2(w2, 1);
    if !(1..=12).contains(&mo) || h > 23 || mi > 59 || s > 59 || d < 1 {
        return None;
    }
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    let mut ms = (days_ym(y, mo) + d - 1) * DAY + (h * 3600 + mi * 60 + s) * 1000;
    if n == 20 {
        // b[19] == 'Z' checked by the branch above (n-1)
        if b[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = num(&b[20..22])?;
        let om = num(&b[23..25])?;
        if oh > 23 || om > 59 {
            return None;
        }
        ms -= sign * (oh * 3_600_000 + om * 60_000);
    }
    Some(ms)
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    let mut sig = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        if sig > 0 || c != b'0' {
            sig += 1;
            if sig > 8 {
                return None;
            }
            v = v * 10 + (c - b'0') as u64;
        }
    }
    if (1..=10_000_000).contains(&v) {
        Some(v as u32)
    } else {
        None
    }
}

#[inline]
fn mkref(b: &[u8]) -> u64 {
    (b.as_ptr() as u64 & PTR_MASK) | ((b.len().min(0xFFFF) as u64) << 48)
}
#[inline]
fn deref(r: u64) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts((r & PTR_MASK) as *const u8, (r >> 48) as usize) }
}

/// The value starting at p (inside a mapped line) up to its delimiter.
#[inline]
fn val_at(p: u64) -> &'static [u8] {
    let s = p as *const u8;
    let mut n = 0usize;
    unsafe {
        loop {
            let c = *s.add(n);
            if c == b',' || c == b'"' || c == b'\n' {
                break;
            }
            n += 1;
        }
        std::slice::from_raw_parts(s, n)
    }
}

// ------------------------------------------------------------------ records
const PTR_MASK: u64 = (1u64 << 48) - 1;

/// One parsed event copy (40 bytes). person == 0 means "invalid copy".
#[derive(Clone, Copy)]
#[repr(C)]
struct ERec {
    eid: u64,
    person: u64,
    nref: u64, // name: pointer (48 bits) | length << 48
    eid2: u32,
    ts: u32,
    ing: u32, // ingest seconds, biased
    pc: u32,  // price (24 bits, > 0 only for purchases) | category << 24
}

/// A surviving event (24 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct SRec {
    person: u64,
    nref: u64,
    ts: u32,
    pc: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    person: u64,
    cidp: u64,
    ts: u32,
    camp: u16,
}

pub struct URow {
    pub name: &'static [u8],
    pub day: i32,
    pub sessions: i64,
    pub events: i64,
    pub rev: i64,
    pub top: i16,
}

struct Ctx {
    base: i64, // window_start - 7 days; all u32 times are ms since base
    win_lo: i64,
    win_hi: i64,
    ck_lo: i64,
    link: SMap, // linked id hash -> name ref of the person
    products: Table,
    cats: Vec<&'static [u8]>,
    unknown_cat: u8,
    cat_order: Vec<u8>,
    camps: Table,
    channels: Vec<&'static [u8]>,
    camp_names: Vec<&'static [u8]>,
    day0: i64,
    ndays: usize,
}

/// u64 key -> u64 value, open addressing, 16-byte slots (one cache line touch).
struct HMap {
    mask: usize,
    slots: &'static mut [(u64, u64)],
}
fn hmap_cap(n: usize) -> usize {
    (n * 2 + 16).next_power_of_two()
}
/// Zeroed memory from the kernel, huge-page advised.
fn zalloc<T>(n: usize) -> &'static mut [T] {
    let bytes = (n * std::mem::size_of::<T>()).max(1);
    let p = unsafe { mmap(std::ptr::null_mut(), bytes, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
    assert!(p as isize != -1 && !p.is_null(), "mmap failed");
    unsafe {
        madvise(p, bytes, 14);
        std::slice::from_raw_parts_mut(p as *mut T, n)
    }
}
impl HMap {
    fn with_slots(slots: &'static mut [(u64, u64)]) -> HMap {
        HMap { mask: slots.len() - 1, slots }
    }
    #[inline]
    fn get(&self, k: u64) -> Option<u64> {
        let mut i = (k as usize) & self.mask;
        loop {
            let (kk, v) = unsafe { *self.slots.get_unchecked(i) };
            if kk == k {
                return Some(v);
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let p = self.slots.as_ptr().add((k as usize) & self.mask) as *const i8;
            core::arch::x86_64::_mm_prefetch(p, core::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = k;
    }
    fn set(&mut self, k: u64, v: u64) {
        let mut i = (k as usize) & self.mask;
        loop {
            let kk = self.slots[i].0;
            if kk == k || kk == 0 {
                self.slots[i] = (k, v);
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// NPART shards of HMap selected by the top 8 bits of the key.
struct SMap {
    shards: Vec<HMap>,
}
impl SMap {
    #[inline]
    fn get(&self, k: u64) -> Option<u64> {
        unsafe { self.shards.get_unchecked((k >> PSHIFT) as usize) }.get(k)
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        unsafe { self.shards.get_unchecked((k >> PSHIFT) as usize) }.prefetch(k)
    }
}

/// Small growable u64 -> u32 map with cheap reset.
struct LMap {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<u32>,
    used: Vec<u32>,
}
impl LMap {
    fn new() -> LMap {
        LMap { mask: 1023, keys: vec![0; 1024], vals: vec![0; 1024], used: Vec::new() }
    }
    fn reset(&mut self, expect: usize) {
        for &u in &self.used {
            self.keys[u as usize] = 0;
        }
        self.used.clear();
        let cap = (expect * 2 + 64).next_power_of_two();
        if self.keys.len() < cap {
            self.keys = vec![0; cap];
            self.vals = vec![0; cap];
        }
        self.mask = cap.min(self.keys.len()) - 1;
    }
    fn grow(&mut self) {
        let cap = (self.mask + 1) * 2;
        let old: Vec<(u64, u32)> = self.used.iter().map(|&u| (self.keys[u as usize], self.vals[u as usize])).collect();
        for &u in &self.used {
            self.keys[u as usize] = 0;
        }
        self.used.clear();
        if self.keys.len() < cap {
            self.keys = vec![0; cap];
            self.vals = vec![0; cap];
        }
        self.mask = cap - 1;
        for (k, v) in old {
            let mut i = (k as usize ^ (k >> 32) as usize) & self.mask;
            while self.keys[i] != 0 {
                i = (i + 1) & self.mask;
            }
            self.keys[i] = k;
            self.vals[i] = v;
            self.used.push(i as u32);
        }
    }
    #[inline]
    fn get_or_insert(&mut self, k: u64, v: u32) -> (u32, bool) {
        if self.used.len() * 2 > self.mask {
            self.grow();
        }
        let mut i = (k as usize ^ (k >> 32) as usize) & self.mask;
        loop {
            let kk = unsafe { *self.keys.get_unchecked(i) };
            if kk == k {
                return (self.vals[i], false);
            }
            if kk == 0 {
                self.keys[i] = k;
                self.vals[i] = v;
                self.used.push(i as u32);
                return (v, true);
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn get(&self, k: u64) -> Option<u32> {
        let mut i = (k as usize ^ (k >> 32) as usize) & self.mask;
        loop {
            let kk = unsafe { *self.keys.get_unchecked(i) };
            if kk == k {
                return Some(self.vals[i]);
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
}

fn read_manifest(data: &Path) -> (i64, i64) {
    let s = fs::read_to_string(data.join("manifest.json")).expect("manifest");
    let get = |key: &str| -> i64 {
        let i = s.find(&format!("\"{}\"", key)).expect("key") + key.len() + 2;
        let rest = &s[i..];
        let a = rest.find('"').unwrap() + 1;
        let b = a + rest[a..].find('"').unwrap();
        parse_ts(rest[a..b].as_bytes()).expect("manifest ts")
    };
    (get("window_start"), get("window_end"))
}

/// Lines of a small single file (header skipped), sequential.
fn small_rows<const K: usize>(path: &Path) -> Vec<[&'static [u8]; K]> {
    let mut v = Vec::new();
    if !path.exists() {
        return v;
    }
    let d = map_file(path);
    for_rows::<K>(d, 0, d.len(), |f| v.push(f));
    v
}

fn par_for(n: usize, nt: usize, f: impl Fn(usize, usize) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for t in 0..nt {
            let next = &next;
            let f = &f;
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                f(t, i);
            });
        }
    });
}

extern "C" {
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

/// NPART append-only buffers backed by one huge-page arena with a fixed
/// capacity per part; overflow spills into ordinary Vecs.
struct Parts<T: Copy> {
    base: *mut T,
    cap: usize,
    len: Vec<usize>,
    over: Vec<Vec<T>>,
}
unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}
impl<T: Copy> Parts<T> {
    fn new(cap: usize) -> Parts<T> {
        let cap = cap.div_ceil(WCN) * WCN;
        let bytes = NPART * cap * std::mem::size_of::<T>();
        let mut base = std::ptr::null_mut::<T>();
        let mut cap = cap;
        if bytes > 0 {
            // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE
            let p = unsafe { mmap(std::ptr::null_mut(), bytes, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
            if p as isize == -1 || p.is_null() {
                cap = 0;
            } else {
                unsafe { madvise(p, bytes, 14) }; // MADV_HUGEPAGE
                base = p as *mut T;
            }
        }
        Parts { base, cap, len: vec![0; NPART], over: (0..NPART).map(|_| Vec::new()).collect() }
    }
    #[inline(always)]
    fn push(&mut self, p: usize, v: T) {
        let l = unsafe { self.len.get_unchecked_mut(p) };
        if *l < self.cap {
            unsafe { self.base.add(p * self.cap + *l).write(v) };
            *l += 1;
        } else {
            unsafe { self.over.get_unchecked_mut(p) }.push(v);
        }
    }
    /// Append WCN elements; whole cache lines go out with streaming stores.
    #[inline(always)]
    fn push_line(&mut self, p: usize, src: &[T; WCN]) {
        let l = unsafe { *self.len.get_unchecked(p) };
        if l + WCN <= self.cap {
            unsafe {
                let dst = self.base.add(p * self.cap + l) as *mut u8;
                stream_copy(dst, src.as_ptr() as *const u8, WCN * std::mem::size_of::<T>());
                *self.len.get_unchecked_mut(p) = l + WCN;
            }
        } else {
            for v in src.iter() {
                self.push(p, *v);
            }
        }
    }
    fn get(&self, p: usize) -> [&[T]; 2] {
        let a: &[T] = if self.len[p] == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.base.add(p * self.cap), self.len[p]) }
        };
        [a, &self.over[p][..]]
    }
}

const WCN: usize = 8;

/// Copy `bytes` (a multiple of 64, dst 64-byte aligned) with non-temporal stores.
#[inline(always)]
unsafe fn stream_copy(dst: *mut u8, src: *const u8, bytes: usize) {
    #[cfg(target_feature = "avx2")]
    {
        use core::arch::x86_64::*;
        let mut o = 0;
        while o < bytes {
            let v = _mm256_loadu_si256(src.add(o) as *const __m256i);
            _mm256_stream_si256(dst.add(o) as *mut __m256i, v);
            o += 32;
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    std::ptr::copy_nonoverlapping(src, dst, bytes);
}

#[inline(always)]
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
}

/// Software write-combining stage in front of a Parts.
struct Wc<T: Copy> {
    stage: Vec<[T; WCN]>,
    cnt: Vec<u8>,
}
impl<T: Copy> Wc<T> {
    fn new() -> Wc<T> {
        Wc { stage: vec![unsafe { std::mem::zeroed() }; NPART], cnt: vec![0; NPART] }
    }
    #[inline(always)]
    fn push(&mut self, parts: &mut Parts<T>, p: usize, v: T) {
        unsafe {
            let c = *self.cnt.get_unchecked(p) as usize;
            let st = self.stage.get_unchecked_mut(p);
            *st.get_unchecked_mut(c) = v;
            if c + 1 == WCN {
                parts.push_line(p, st);
                *self.cnt.get_unchecked_mut(p) = 0;
            } else {
                *self.cnt.get_unchecked_mut(p) = (c + 1) as u8;
            }
        }
    }
    fn finish(&mut self, parts: &mut Parts<T>) {
        for p in 0..NPART {
            for i in 0..self.cnt[p] as usize {
                parts.push(p, self.stage[p][i]);
            }
            self.cnt[p] = 0;
        }
        sfence();
    }
}

struct SyncCell<T>(std::cell::UnsafeCell<T>);
unsafe impl<T: Send> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    fn new(v: T) -> Self {
        SyncCell(std::cell::UnsafeCell::new(v))
    }
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut T {
        &mut *self.0.get()
    }
    fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

fn main() {
    let t0 = std::time::Instant::now();
    let verbose = std::env::var("VERBOSE").is_ok();
    let lap = |what: &str| {
        if verbose {
            eprintln!("{:8.3}s {}", t0.elapsed().as_secs_f64(), what);
        }
    };
    let args: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let _ = args;
    let nt: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        .clamp(1, 32);

    let (win_lo, win_hi) = read_manifest(&data);
    let base = win_lo - WEEK;

    // ---------------------------------------------------------------- links
    let link_files = map_all(&csv_files(&data.join("links")), nt);
    let lunits: Vec<(usize, usize, usize)> = {
        let mut u = Vec::new();
        for (fi, d) in link_files.iter().enumerate() {
            let mut s = 0;
            loop {
                let e = (s + (1 << 20)).min(d.len());
                u.push((fi, s, e));
                if e >= d.len() {
                    break;
                }
                s = e;
            }
        }
        u
    };
    type Pair = (u64, u64, u64, u64); // (hash a, hash b, ref a, ref b)
    let lbufs: Vec<SyncCell<Vec<Pair>>> = (0..nt).map(|_| SyncCell::new(Vec::new())).collect();
    par_for(lunits.len(), nt, |t, u| {
        let (fi, s, e) = lunits[u];
        let b = unsafe { lbufs[t].get() };
        for_rows::<2>(link_files[fi], s, e, |[a, c]| {
            if !a.is_empty() && !c.is_empty() {
                b.push((hash(a), hash(c), mkref(a), mkref(c)));
            }
        });
    });
    let mut pairs: Vec<Pair> = Vec::new();
    for c in lbufs {
        pairs.extend_from_slice(&c.into_inner());
    }
    let ne = pairs.len();
    // endpoints -> buckets by top hash bits: (hash, ref, endpoint index)
    let ep_cap = (ne * 2) / (nt * NPART) + (ne * 2) / (nt * NPART * 5) + 64;
    let epb: Vec<SyncCell<Parts<(u64, u64, u32)>>> = (0..nt).map(|_| SyncCell::new(Parts::new(ep_cap))).collect();
    const LCH: usize = 1 << 15;
    par_for(ne.div_ceil(LCH), nt, |t, c| {
        let b = unsafe { epb[t].get() };
        for e in c * LCH..((c + 1) * LCH).min(ne) {
            let (ha, hb, ra, rb) = pairs[e];
            b.push((ha >> PSHIFT) as usize, (ha, ra, (e * 2) as u32));
            b.push((hb >> PSHIFT) as usize, (hb, rb, (e * 2 + 1) as u32));
        }
    });
    let epb: Vec<Parts<(u64, u64, u32)>> = epb.into_iter().map(|c| c.into_inner()).collect();
    // per bucket: unique ids in first-seen order, and the local id of every endpoint
    let buids: Vec<SyncCell<(Vec<(u64, u64)>, Vec<u32>)>> = (0..NPART).map(|_| SyncCell::new((Vec::new(), Vec::new()))).collect();
    let lmaps: Vec<SyncCell<LMap>> = (0..nt).map(|_| SyncCell::new(LMap::new())).collect();
    par_for(NPART, nt, |t, bk| {
        let m = unsafe { lmaps[t].get() };
        let (uniq, loc) = unsafe { buids[bk].get() };
        let n: usize = epb.iter().map(|b| b.get(bk).iter().map(|s| s.len()).sum::<usize>()).sum();
        m.reset(n);
        loc.reserve(n);
        for b in &epb {
            for sl in b.get(bk) {
                for &(h, r, _) in sl.iter() {
                    let (id, new) = m.get_or_insert(h, uniq.len() as u32);
                    if new {
                        uniq.push((h, r));
                    }
                    loc.push(id);
                }
            }
        }
    });
    let mut bbase = vec![0u32; NPART + 1];
    for bk in 0..NPART {
        bbase[bk + 1] = bbase[bk] + unsafe { buids[bk].get() }.0.len() as u32;
    }
    let nid = bbase[NPART] as usize;
    let ends: SyncCell<Vec<u32>> = SyncCell::new(vec![0u32; ne * 2]);
    par_for(NPART, nt, |_, bk| {
        let (_, loc) = unsafe { buids[bk].get() };
        let endp = unsafe { ends.get() }.as_mut_ptr() as usize as *mut u32;
        let mut k = 0;
        for b in &epb {
            for sl in b.get(bk) {
                for &(_, _, ei) in sl.iter() {
                    unsafe { *endp.add(ei as usize) = bbase[bk] + loc[k] };
                    k += 1;
                }
            }
        }
    });
    drop(epb);
    let ends = ends.into_inner();
    let mut parent: Vec<u32> = (0..nid as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for e in 0..ne {
        let ra = find(&mut parent, ends[2 * e]);
        let rb = find(&mut parent, ends[2 * e + 1]);
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    for i in 0..nid {
        let r = find(&mut parent, i as u32);
        parent[i] = r;
    }
    let root = parent;
    let idref = |i: u32| -> u64 {
        let bk = bbase.partition_point(|&x| x <= i) - 1;
        unsafe { buids[bk].get() }.0[(i - bbase[bk]) as usize].1
    };
    let best: Vec<std::sync::atomic::AtomicU32> = (0..nid as u32).map(std::sync::atomic::AtomicU32::new).collect();
    par_for(NPART, nt, |_, bk| {
        let (uniq, _) = unsafe { buids[bk].get() };
        for (j, &(_, r)) in uniq.iter().enumerate() {
            let i = bbase[bk] + j as u32;
            let rt = root[i as usize] as usize;
            let name = deref(r);
            let mut cur = best[rt].load(Ordering::Relaxed);
            while name < deref(idref(cur)) {
                match best[rt].compare_exchange_weak(cur, i, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(x) => cur = x,
                }
            }
        }
    });
    let caps: Vec<usize> = (0..NPART).map(|bk| hmap_cap(unsafe { buids[bk].get() }.0.len())).collect();
    let mut arena: &'static mut [(u64, u64)] = zalloc(caps.iter().sum());
    let mut slices: Vec<SyncCell<&'static mut [(u64, u64)]>> = Vec::with_capacity(NPART);
    for &c in &caps {
        let (a, b) = std::mem::take(&mut arena).split_at_mut(c);
        slices.push(SyncCell::new(a));
        arena = b;
    }
    let shards: Vec<SyncCell<Option<HMap>>> = (0..NPART).map(|_| SyncCell::new(None)).collect();
    par_for(NPART, nt, |_, bk| {
        let (uniq, _) = unsafe { buids[bk].get() };
        let sl: &'static mut [(u64, u64)] = std::mem::take(unsafe { slices[bk].get() });
        let mut m = HMap::with_slots(sl);
        for (j, &(h, _)) in uniq.iter().enumerate() {
            let i = bbase[bk] as usize + j;
            let b = best[root[i] as usize].load(Ordering::Relaxed);
            m.set(h, idref(b));
        }
        unsafe { *shards[bk].get() = Some(m) };
    });
    let person_map = SMap { shards: shards.into_iter().map(|c| c.into_inner().unwrap()).collect() };
    lap("links");

    // ---------------------------------------------------------------- reference data
    let mut cats: Vec<&'static [u8]> = Vec::new();
    let mut cat_tab = Table::new(64);
    let prod_rows = small_rows::<2>(&data.join("products.csv"));
    let mut products = Table::new(prod_rows.len());
    for [p, c] in &prod_rows {
        if p.is_empty() {
            continue;
        }
        let n = cats.len() as u32;
        let ci = cat_tab.insert(c, n);
        if ci == n {
            cats.push(c);
        }
        products.insert(p, ci);
    }
    let unknown_cat = cats.len() as u8;
    cats.push(b"unknown");
    assert!(cats.len() < 250);
    let mut cat_order: Vec<u8> = (0..cats.len() as u8).collect();
    cat_order.sort_by(|&a, &b| cats[a as usize].cmp(cats[b as usize]));
    let camp_rows = small_rows::<2>(&data.join("campaigns.csv"));
    let mut camps = Table::new(camp_rows.len());
    let mut channels = Vec::new();
    let mut camp_names = Vec::new();
    for [c, ch] in &camp_rows {
        if c.is_empty() {
            continue;
        }
        let n = camp_names.len() as u32;
        if camps.insert(c, n) == n {
            camp_names.push(*c);
            channels.push(*ch);
        }
    }
    let day0 = win_lo.div_euclid(DAY);
    let ndays = ((win_hi - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    let ctx = Ctx {
        base,
        win_lo,
        win_hi,
        ck_lo: win_lo - WEEK,
        link: person_map,
        products,
        cats,
        unknown_cat,
        cat_order,
        camps,
        channels,
        camp_names,
        day0,
        ndays,
    };
    let ncamp = ctx.camp_names.len();
    lap("reference");

    // ---------------------------------------------------------------- parse events + clicks
    let ev_paths = csv_files(&data.join("events"));
    let ck_paths = csv_files(&data.join("clicks"));
    let ev_files = map_all(&ev_paths, nt);
    let ck_files = map_all(&ck_paths, nt);
    // work units: (is_click, file, start, end), largest first
    let mut units: Vec<(bool, usize, usize, usize)> = Vec::new();
    for (kind, files) in [(false, &ev_files), (true, &ck_files)] {
        for (fi, d) in files.iter().enumerate() {
            let mut s = 0;
            loop {
                let e = (s + CHUNK).min(d.len());
                units.push((kind, fi, s, e));
                if e >= d.len() {
                    break;
                }
                s = e;
            }
        }
    }
    lap("mapped");

    type EBufs = Parts<ERec>;
    type CBufs = Parts<CRec>;
    let ev_bytes: usize = ev_files.iter().map(|d| d.len()).sum();
    let ck_bytes: usize = ck_files.iter().map(|d| d.len()).sum();
    let per = |rows: usize| rows / (nt * NPART) + rows / (nt * NPART * 10) + 64;
    let ecap = per(ev_bytes / 70);
    let ccap = per(ck_bytes / 44);
    let ebufs: Vec<SyncCell<EBufs>> = (0..nt).map(|_| SyncCell::new(Parts::new(ecap))).collect();
    let ewc: Vec<SyncCell<Wc<ERec>>> = (0..nt).map(|_| SyncCell::new(Wc::new())).collect();
    let cbufs: Vec<SyncCell<CBufs>> = (0..nt).map(|_| SyncCell::new(Parts::new(ccap))).collect();
    let cwc: Vec<SyncCell<Wc<CRec>>> = (0..nt).map(|_| SyncCell::new(Wc::new())).collect();
    let ing_bias: i64 = (base.div_euclid(1000)) - (1i64 << 31);
    par_for(units.len(), nt, |t, u| {
        let (is_click, fi, s, e) = units[u];
        if !is_click {
            let eb = unsafe { ebufs[t].get() };
            let wc = unsafe { ewc[t].get() };
            let d = ev_files[fi];
            const B: usize = 16;
            let mut batch: [ERec; B] = unsafe { std::mem::zeroed() };
            let mut nb = 0usize;
            let link = &ctx.link;
            let flush = |batch: &mut [ERec; B], nb: usize, eb: &mut EBufs, wc: &mut Wc<ERec>| {
                for r in batch[..nb].iter_mut() {
                    if let Some(nr) = link.get(r.person) {
                        r.nref = nr;
                        r.person = fmix(nr);
                    }
                    wc.push(eb, (r.eid >> PSHIFT) as usize, *r);
                }
            };
            for_rows::<7>(d, s, e, |[eid, uid, ts, ing, typ, pid, price]| {
                if eid.is_empty() {
                    return;
                }
                let ingv = parse_ts(ing).unwrap_or(i64::MIN);
                let (eh, eh2) = hash2(eid);
                let mut r = ERec {
                    eid: eh,
                    person: 0,
                    nref: 0,
                    eid2: eh2,
                    ts: 0,
                    ing: (ingv.div_euclid(1000).saturating_sub(ing_bias)).clamp(0, u32::MAX as i64) as u32,
                    pc: 0,
                };
                let mut cat: u32 = 0;
                let kind = match typ {
                    b"view" => 1u8,
                    b"cart" => 2,
                    b"purchase" => 3,
                    _ => 0,
                };
                let tsv = parse_ts(ts);
                let mut ok = kind != 0 && !uid.is_empty() && !pid.is_empty();
                if ok {
                    match tsv {
                        Some(v) if v >= ctx.win_lo && v < ctx.win_hi && ingv.saturating_sub(v) <= DAY => {
                            r.ts = (v - ctx.base) as u32;
                        }
                        _ => ok = false,
                    }
                }
                if ok && kind == 3 {
                    match parse_price(price) {
                        Some(p) => {
                            r.pc = p;
                            cat = ctx.products.get(pid).unwrap_or(ctx.unknown_cat as u32);
                        }
                        None => ok = false,
                    }
                }
                if ok {
                    let h = hash(uid);
                    link.prefetch(h);
                    r.person = h;
                    r.nref = mkref(uid);
                    r.pc |= cat << 24;
                    batch[nb] = r;
                    nb += 1;
                    if nb == B {
                        flush(&mut batch, nb, eb, wc);
                        nb = 0;
                    }
                } else {
                    wc.push(eb, (r.eid >> PSHIFT) as usize, r);
                }
            });
            flush(&mut batch, nb, eb, wc);
            sfence();
        } else {
            let cb = unsafe { cbufs[t].get() };
            let cw = unsafe { cwc[t].get() };
            let d = ck_files[fi];
            const B: usize = 16;
            let mut batch: [CRec; B] = unsafe { std::mem::zeroed() };
            let mut nb = 0usize;
            let link = &ctx.link;
            let flush = |batch: &mut [CRec; B], nb: usize, cb: &mut CBufs, cw: &mut Wc<CRec>| {
                for r in batch[..nb].iter_mut() {
                    if let Some(nr) = link.get(r.person) {
                        r.person = fmix(nr);
                    }
                    cw.push(cb, (r.person >> PSHIFT) as usize, *r);
                }
            };
            for_rows::<4>(d, s, e, |[cid, uid, camp, ts]| {
                if cid.is_empty() || uid.is_empty() {
                    return;
                }
                let Some(v) = parse_ts(ts) else { return };
                if v < ctx.ck_lo || v >= ctx.win_hi {
                    return;
                }
                let Some(ci) = ctx.camps.get(camp) else { return };
                let h = hash(uid);
                link.prefetch(h);
                batch[nb] = CRec { person: h, cidp: cid.as_ptr() as u64, ts: (v - ctx.base) as u32, camp: ci as u16 };
                nb += 1;
                if nb == B {
                    flush(&mut batch, nb, cb, cw);
                    nb = 0;
                }
            });
            flush(&mut batch, nb, cb, cw);
            sfence();
        }
    });
    for (wc, eb) in ewc.iter().zip(ebufs.iter()) {
        unsafe { wc.get().finish(eb.get()) };
    }
    let ebufs: Vec<EBufs> = ebufs.into_iter().map(|c| c.into_inner()).collect();
    for (wc, cb) in cwc.iter().zip(cbufs.iter()) {
        unsafe { wc.get().finish(cb.get()) };
    }
    let cbufs: Vec<CBufs> = cbufs.into_iter().map(|c| c.into_inner()).collect();
    lap("parsed");

    // ---------------------------------------------------------------- dedup events
    let scap = per(ev_bytes / 70);
    let sbufs: Vec<SyncCell<Parts<SRec>>> = (0..nt).map(|_| SyncCell::new(Parts::new(scap))).collect();
    let swc: Vec<SyncCell<Wc<SRec>>> = (0..nt).map(|_| SyncCell::new(Wc::new())).collect();
    let dtabs: Vec<SyncCell<Vec<(u64, u64)>>> = (0..nt).map(|_| SyncCell::new(Vec::new())).collect();
    par_for(NPART, nt, |t, p| {
        let mut segs: Vec<&[ERec]> = Vec::with_capacity(2 * nt);
        let mut n = 0;
        for b in &ebufs {
            for sl in b.get(p) {
                if !sl.is_empty() {
                    n += sl.len();
                    segs.push(sl);
                }
            }
        }
        let cap = (n * 2 + 16).next_power_of_two();
        let mask = cap - 1;
        let tab = unsafe { dtabs[t].get() };
        if tab.len() < cap {
            tab.resize(cap, (0, 0));
        }
        // slot: (eid, eid2 << 32 | seg << 26 | (idx + 1)), 0 = empty
        for (si, seg) in segs.iter().enumerate() {
            for (ri, r) in seg.iter().enumerate() {
                let tag = ((r.eid2 as u64) << 32) | ((si as u64) << 26) | (ri as u64 + 1);
                let mut h = (r.eid as usize ^ (r.eid >> 29) as usize) & mask;
                loop {
                    let sl = unsafe { tab.get_unchecked_mut(h) };
                    if sl.1 == 0 {
                        *sl = (r.eid, tag);
                        break;
                    }
                    if sl.0 == r.eid && (sl.1 >> 32) as u32 == r.eid2 {
                        let os = ((sl.1 >> 26) & 63) as usize;
                        let oi = ((sl.1 & ((1 << 26) - 1)) - 1) as usize;
                        if r.ing < segs[os][oi].ing {
                            sl.1 = tag;
                        }
                        break;
                    }
                    h = (h + 1) & mask;
                }
            }
        }
        let sb = unsafe { sbufs[t].get() };
        let sw = unsafe { swc[t].get() };
        for sl in tab[..cap].iter_mut() {
            if sl.1 == 0 {
                continue;
            }
            let os = ((sl.1 >> 26) & 63) as usize;
            let oi = ((sl.1 & ((1 << 26) - 1)) - 1) as usize;
            *sl = (0, 0);
            let r = &segs[os][oi];
            if r.person != 0 {
                sw.push(sb, (r.person >> PSHIFT) as usize, SRec { person: r.person, nref: r.nref, ts: r.ts, pc: r.pc });
            }
        }
        sfence();
    });
    drop(ebufs);
    for (wc, sb) in swc.iter().zip(sbufs.iter()) {
        unsafe { wc.get().finish(sb.get()) };
    }
    let sbufs: Vec<Parts<SRec>> = sbufs.into_iter().map(|c| c.into_inner()).collect();
    lap("dedup");

    // ---------------------------------------------------------------- sessions, credit, assists
    let ncat = ctx.cats.len();
    let organic = ncamp;
    let urows: Vec<SyncCell<Vec<URow>>> = (0..nt).map(|_| SyncCell::new(Vec::new())).collect();
    let cagg: Vec<SyncCell<Vec<[i64; 4]>>> =
        (0..nt).map(|_| SyncCell::new(vec![[0i64; 4]; (ncamp + 1) * ctx.ndays])).collect();
    struct GS {
        map: LMap,
        tmp: Vec<u32>,
        off: Vec<u32>,
        pos: Vec<u32>,
        ev: Vec<SRec>,
        ctmp: Vec<u32>,
        coff: Vec<u32>,
        cpos: Vec<u32>,
        ck: Vec<CRec>,
    }
    let gath: Vec<SyncCell<GS>> = (0..nt)
        .map(|_| {
            SyncCell::new(GS {
                map: LMap::new(),
                tmp: Vec::new(),
                off: Vec::new(),
                pos: Vec::new(),
                ev: Vec::new(),
                ctmp: Vec::new(),
                coff: Vec::new(),
                cpos: Vec::new(),
                ck: Vec::new(),
            })
        })
        .collect();
    let scratch: Vec<SyncCell<(Vec<u32>, Vec<u32>, Vec<u16>)>> =
        (0..nt).map(|_| SyncCell::new((vec![0; ncamp + 1], vec![0; ncamp + 1], Vec::new()))).collect();
    par_for(NPART, nt, |t, p| {
        let g = unsafe { gath[t].get() };
        let nev: usize = sbufs.iter().map(|b| b.get(p).iter().map(|s| s.len()).sum::<usize>()).sum();
        g.map.reset(nev / 16 + 64);
        g.tmp.clear();
        g.off.clear();
        // group events by person (counting sort)
        for b in &sbufs {
            for sl in b.get(p) {
                for r in sl.iter() {
                    let (id, new) = g.map.get_or_insert(r.person, g.off.len() as u32);
                    if new {
                        g.off.push(0);
                    }
                    g.off[id as usize] += 1;
                    g.tmp.push(id);
                }
            }
        }
        let np = g.off.len();
        let mut acc = 0u32;
        g.pos.clear();
        for x in g.off.iter_mut() {
            let c = *x;
            *x = acc;
            g.pos.push(acc);
            acc += c;
        }
        g.off.push(acc);
        g.ev.clear();
        g.ev.reserve(nev);
        unsafe { g.ev.set_len(nev) };
        let mut k = 0;
        for b in &sbufs {
            for sl in b.get(p) {
                for r in sl.iter() {
                    let id = g.tmp[k] as usize;
                    k += 1;
                    let q = g.pos[id] as usize;
                    g.pos[id] += 1;
                    unsafe { *g.ev.get_unchecked_mut(q) = *r };
                }
            }
        }
        // clicks of persons that have events, grouped the same way
        g.ctmp.clear();
        g.coff.clear();
        g.coff.resize(np, 0);
        for b in &cbufs {
            for sl in b.get(p) {
                for c in sl.iter() {
                    match g.map.get(c.person) {
                        Some(id) => {
                            g.coff[id as usize] += 1;
                            g.ctmp.push(id);
                        }
                        None => g.ctmp.push(u32::MAX),
                    }
                }
            }
        }
        let mut acc = 0u32;
        g.cpos.clear();
        for x in g.coff.iter_mut() {
            let c = *x;
            *x = acc;
            g.cpos.push(acc);
            acc += c;
        }
        g.coff.push(acc);
        g.ck.clear();
        g.ck.reserve(acc as usize);
        unsafe { g.ck.set_len(acc as usize) };
        let mut k = 0;
        for b in &cbufs {
            for sl in b.get(p) {
                for c in sl.iter() {
                    let id = g.ctmp[k];
                    k += 1;
                    if id != u32::MAX {
                        let q = g.cpos[id as usize] as usize;
                        g.cpos[id as usize] += 1;
                        unsafe { *g.ck.get_unchecked_mut(q) = *c };
                    }
                }
            }
        }
        let rows = unsafe { urows[t].get() };
        let ca = unsafe { cagg[t].get() };
        let (cnt, pos, active) = unsafe { scratch[t].get() };
        let mut catsum = vec![0i64; ncat];
        for gi in 0..np {
            let es = &mut g.ev[g.off[gi] as usize..g.off[gi + 1] as usize];
            es.sort_unstable_by_key(|r| r.ts);
            let cl = &mut g.ck[g.coff[gi] as usize..g.coff[gi + 1] as usize];
            if cl.len() > 1 {
                cl.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| val_at(a.cidp).cmp(val_at(b.cidp))));
            }
            let es = &*es;
            let cl = &*cl;
            let name = deref(es[0].nref);

            let (mut lo, mut hi) = (0usize, 0usize);
            let mut cur_day: i64 = i64::MIN;
            let (mut ds, mut de, mut dr) = (0i64, 0i64, 0i64);
            let mut k = 0;
            let flush = |rows: &mut Vec<URow>, day: i64, ds: i64, de: i64, dr: i64, catsum: &mut [i64]| {
                let mut top: i16 = -1;
                let mut bestv = 0i64;
                for &c in &ctx.cat_order {
                    let v = catsum[c as usize];
                    if v > bestv {
                        bestv = v;
                        top = c as i16;
                    }
                }
                for x in catsum.iter_mut() {
                    *x = 0;
                }
                rows.push(URow { name, day: day as i32, sessions: ds, events: de, rev: dr, top });
            };
            while k < es.len() {
                let start = es[k].ts;
                let day = (ctx.base + start as i64).div_euclid(DAY);
                if day != cur_day {
                    if cur_day != i64::MIN {
                        flush(rows, cur_day, ds, de, dr, &mut catsum);
                    }
                    cur_day = day;
                    ds = 0;
                    de = 0;
                    dr = 0;
                }
                let mut last = start;
                let mut nev = 0i64;
                let mut rev = 0i64;
                let mut has_p = false;
                while k < es.len() && es[k].ts - last <= GAP {
                    let r = &es[k];
                    last = r.ts;
                    nev += 1;
                    let price = r.pc & 0x00FF_FFFF;
                    if price != 0 {
                        rev += price as i64;
                        has_p = true;
                        catsum[(r.pc >> 24) as usize] += price as i64;
                    }
                    k += 1;
                }
                ds += 1;
                de += nev;
                dr += rev;
                // clicks: window [start - 7d, start]
                while hi < cl.len() && cl[hi].ts <= start {
                    let c = cl[hi].camp as usize;
                    if cnt[c] == 0 {
                        pos[c] = active.len() as u32;
                        active.push(c as u16);
                    }
                    cnt[c] += 1;
                    hi += 1;
                }
                while lo < hi && (cl[lo].ts as i64) + WEEK < start as i64 {
                    let c = cl[lo].camp as usize;
                    cnt[c] -= 1;
                    if cnt[c] == 0 {
                        let pi = pos[c] as usize;
                        let lastc = *active.last().unwrap();
                        active[pi] = lastc;
                        pos[lastc as usize] = pi as u32;
                        active.pop();
                    }
                    lo += 1;
                }
                let didx = (day - ctx.day0) as usize;
                let camp = if hi > lo {
                    let mut idx = hi - 1;
                    while idx > lo && cl[idx - 1].ts == cl[hi - 1].ts {
                        idx -= 1;
                    }
                    cl[idx].camp as usize
                } else {
                    organic
                };
                let a = &mut ca[camp * ctx.ndays + didx];
                a[0] += 1;
                a[1] += has_p as i64;
                a[2] += rev;
                for &c in active.iter() {
                    ca[c as usize * ctx.ndays + didx][3] += 1;
                }
            }
            flush(rows, cur_day, ds, de, dr, &mut catsum);
            for &c in active.iter() {
                cnt[c as usize] = 0;
            }
            active.clear();
        }
    });
    drop(sbufs);
    lap("sessions");

    // ---------------------------------------------------------------- outputs
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    fs::create_dir_all(&ud).expect("mkdir");
    fs::create_dir_all(&cd).expect("mkdir");
    let urows: Vec<Vec<URow>> = urows.into_iter().map(|c| c.into_inner()).collect();
    let any = urows.iter().any(|r| !r.is_empty());
    std::thread::scope(|s| {
        for (t, rows) in urows.iter().enumerate() {
            if rows.is_empty() && (any || t > 0) {
                continue;
            }
            let path = ud.join(format!("part-{:03}.parquet", t));
            let cats = &ctx.cats;
            s.spawn(move || parquet::write_user_daily(&path, rows, cats));
        }
        // campaign_daily
        let mut tot = vec![[0i64; 4]; (ncamp + 1) * ctx.ndays];
        for c in cagg {
            for (x, y) in tot.iter_mut().zip(c.into_inner().iter()) {
                for q in 0..4 {
                    x[q] += y[q];
                }
            }
        }
        let mut crows = Vec::new();
        for camp in 0..=ncamp {
            for di in 0..ctx.ndays {
                let a = tot[camp * ctx.ndays + di];
                if a[0] > 0 || a[3] > 0 {
                    let (cid, ch): (&[u8], &[u8]) = if camp == organic {
                        (b"organic", b"organic")
                    } else {
                        (ctx.camp_names[camp], ctx.channels[camp])
                    };
                    crows.push(parquet::CRow { campaign: cid, channel: ch, day: (ctx.day0 + di as i64) as i32, v: a });
                }
            }
        }
        parquet::write_campaign_daily(&cd.join("part-000.parquet"), &crows);
    });
    lap("written");
}

#[cfg(test)]
mod difftest {
    use super::*;
    #[inline]
    fn ref_days(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }
    
    #[inline]
    fn ref_dig(c: u8) -> Option<i64> {
        if c.is_ascii_digit() {
            Some((c - b'0') as i64)
        } else {
            None
        }
    }
    
    #[inline]
    fn ref_num(b: &[u8]) -> Option<i64> {
        let mut v = 0i64;
        for &c in b {
            v = v * 10 + ref_dig(c)?;
        }
        Some(v)
    }
    
    /// Epoch ms, or None for a bad timestamp. Huge F3 values map to i64::MAX.
    #[inline]
    fn ref_parse_ts(b: &[u8]) -> Option<i64> {
        let n = b.len();
        if n == 0 {
            return None;
        }
        if !((n == 20 || n == 25) && b[4] == b'-') {
            // F3: digits only
            let mut v: i64 = 0;
            let mut sig = 0;
            for &c in b {
                if !c.is_ascii_digit() {
                    return None;
                }
                if sig > 0 || c != b'0' {
                    sig += 1;
                    if sig > 18 {
                        v = i64::MAX; // out of any window
                        continue;
                    }
                    v = v * 10 + (c - b'0') as i64;
                }
            }
            return Some(v);
        }
        if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let y = ref_num(&b[0..4])?;
        let mo = ref_num(&b[5..7])?;
        let d = ref_num(&b[8..10])?;
        let h = ref_num(&b[11..13])?;
        let mi = ref_num(&b[14..16])?;
        let s = ref_num(&b[17..19])?;
        if !(1..=12).contains(&mo) || h > 23 || mi > 59 || s > 59 || d < 1 {
            return None;
        }
        let dim = match mo {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ => {
                if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                    29
                } else {
                    28
                }
            }
        };
        if d > dim {
            return None;
        }
        let mut ms = ref_days(y, mo, d) * DAY + (h * 3600 + mi * 60 + s) * 1000;
        if n == 20 {
            // b[19] == 'Z' checked by the branch above (n-1)
            if b[19] != b'Z' {
                return None;
            }
        } else {
            let sign = match b[19] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            if b[22] != b':' {
                return None;
            }
            let oh = ref_num(&b[20..22])?;
            let om = ref_num(&b[23..25])?;
            if oh > 23 || om > 59 {
                return None;
            }
            ms -= sign * (oh * 3_600_000 + om * 60_000);
        }
        Some(ms)
    }
    

    #[test]
    fn ts_matches_reference() {
        let mut seed: u64 = 12345;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let alphabet: &[u8] = b"0123456789-T:Z+ .zt";
        let templates: [&[u8]; 4] = [b"2026-10-03T23:58:10Z", b"2026-10-03T19:58:10-04:00", b"1791071890000", b"0001791071890000"];
        let mut n = 0;
        for _ in 0..3_000_000 {
            let t = templates[(rnd() % 4) as usize];
            let mut v = t.to_vec();
            let muts = rnd() % 4;
            for _ in 0..muts {
                let i = (rnd() as usize) % v.len();
                match rnd() % 4 {
                    0 => v[i] = alphabet[(rnd() as usize) % alphabet.len()],
                    1 => v[i] = b'0' + (rnd() % 10) as u8,
                    2 => { if v.len() < 30 { v.insert(i, b'0' + (rnd() % 10) as u8); } }
                    _ => { if v.len() > 1 { v.remove(i); } }
                }
            }
            let a = parse_ts(&v);
            let b = ref_parse_ts(&v);
            assert_eq!(a, b, "{:?}", String::from_utf8_lossy(&v));
            n += a.is_some() as usize;
        }
        // exhaustive-ish calendar sweep
        for y in [1999i64, 2000, 2024, 2025, 2026, 2027, 2028, 2100] {
            for m in 0..=13 { for d in 0..=32 { for h in [0, 23, 24] {
                let s = format!("{:04}-{:02}-{:02}T{:02}:59:59Z", y, m, d, h);
                assert_eq!(parse_ts(s.as_bytes()), ref_parse_ts(s.as_bytes()), "{}", s);
                let s = format!("{:04}-{:02}-{:02}T{:02}:00:00+23:59", y, m, d, h);
                assert_eq!(parse_ts(s.as_bytes()), ref_parse_ts(s.as_bytes()), "{}", s);
            }}}
        }
        for len in 1..=22 { for lead in 0..len { 
            let mut s = vec![b'0'; lead]; while s.len() < len { s.push(b'1' + (s.len() % 9) as u8); }
            assert_eq!(parse_ts(&s), ref_parse_ts(&s), "{:?}", String::from_utf8_lossy(&s));
        }}
        assert!(n > 1000);
    }
}
