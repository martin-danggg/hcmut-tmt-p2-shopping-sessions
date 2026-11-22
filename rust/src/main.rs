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
const NPART: usize = 256;
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
    let mut out: [&'static [u8]; K] = [&d[0..0]; K];
    let mut k = 0usize;
    let mut st = p0;
    let mut q = 0u32;
    let mut bad = false;
    let mut tail: [u8; 64];
    let mut i = p0;
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
            if mn & bit != 0 {
                let le = j.min(n);
                if !bad && k + 1 == K {
                    if let Some(x) = fin(d, st, le, q) {
                        out[k] = x;
                        f(out);
                    }
                }
                st = j + 1;
                if st >= end {
                    return;
                }
                k = 0;
                q = 0;
                bad = false;
            } else if mq & bit != 0 {
                q += 1;
            } else {
                if !bad {
                    if k + 1 >= K {
                        bad = true;
                    } else {
                        match fin(d, st, j, q) {
                            Some(x) => out[k] = x,
                            None => bad = true,
                        }
                        k += 1;
                    }
                }
                st = j + 1;
                q = 0;
            }
            m &= m - 1;
        }
        i += 64;
    }
}

#[inline]
fn fin(line: &'static [u8], st: usize, en: usize, q: u32) -> Option<&'static [u8]> {
    if q == 0 {
        return Some(&line[st..en]);
    }
    if q == 2 && en - st >= 2 && line[st] == b'"' && line[en - 1] == b'"' {
        return Some(&line[st + 1..en - 1]);
    }
    None
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

/// Epoch ms, or None for a bad timestamp. Huge F3 values map to i64::MAX.
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
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
    let y = num(&b[0..4])?;
    let mo = num(&b[5..7])?;
    let d = num(&b[8..10])?;
    let h = num(&b[11..13])?;
    let mi = num(&b[14..16])?;
    let s = num(&b[17..19])?;
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
    let mut ms = days_from_civil(y, mo, d) * DAY + (h * 3600 + mi * 60 + s) * 1000;
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
#[derive(Clone, Copy)]
#[repr(C)]
struct ERec {
    eid: u64,
    ing: i64,
    person: u64,
    uidp: u64,
    eid2: u32,
    ts: u32,
    price: u32,
    kind: u8, // 0 invalid, 1 view, 2 cart, 3 purchase
    cat: u8,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SRec {
    person: u64,
    uidp: u64,
    ts: u32,
    price: u32,
    kind: u8,
    cat: u8,
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
    link: HMap,           // linked id hash -> person hash
    linked_names: Table2, // person hash -> name
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
    slots: Vec<(u64, u64)>,
}
impl HMap {
    fn new(n: usize) -> HMap {
        let cap = (n * 2 + 16).next_power_of_two();
        HMap { mask: cap - 1, slots: vec![(0, 0); cap] }
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
    /// insert if absent; returns stored value
    fn insert(&mut self, k: u64, v: u64) -> u64 {
        let mut i = (k as usize) & self.mask;
        loop {
            let (kk, vv) = self.slots[i];
            if kk == k {
                return vv;
            }
            if kk == 0 {
                self.slots[i] = (k, v);
                return v;
            }
            i = (i + 1) & self.mask;
        }
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

/// u64 key -> name (for linked persons).
struct Table2 {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<&'static [u8]>,
}
impl Table2 {
    fn new(n: usize) -> Table2 {
        let cap = (n * 2 + 16).next_power_of_two();
        Table2 { mask: cap - 1, keys: vec![0; cap], vals: vec![&[][..]; cap] }
    }
    fn insert(&mut self, k: u64, v: &'static [u8]) {
        let mut i = (k as usize) & self.mask;
        loop {
            if self.keys[i] == 0 {
                self.keys[i] = k;
                self.vals[i] = v;
                return;
            }
            if self.keys[i] == k {
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn get(&self, k: u64) -> Option<&'static [u8]> {
        let mut i = (k as usize) & self.mask;
        loop {
            if self.keys[i] == 0 {
                return None;
            }
            if self.keys[i] == k {
                return Some(self.vals[i]);
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
        .max(1);

    let (win_lo, win_hi) = read_manifest(&data);
    let base = win_lo - WEEK;

    // ---------------------------------------------------------------- links
    let link_files = map_all(&csv_files(&data.join("links")), nt);
    let lunits: Vec<(usize, usize, usize)> = {
        let mut u = Vec::new();
        for (fi, d) in link_files.iter().enumerate() {
            let mut s = 0;
            loop {
                let e = (s + CHUNK).min(d.len());
                u.push((fi, s, e));
                if e >= d.len() {
                    break;
                }
                s = e;
            }
        }
        u
    };
    let lbufs: Vec<SyncCell<Vec<(u64, u64, &'static [u8], &'static [u8])>>> =
        (0..nt).map(|_| SyncCell::new(Vec::new())).collect();
    par_for(lunits.len(), nt, |t, u| {
        let (fi, s, e) = lunits[u];
        let b = unsafe { lbufs[t].get() };
        for_rows::<2>(link_files[fi], s, e, |[a, c]| {
            if !a.is_empty() && !c.is_empty() {
                b.push((hash(a), hash(c), a, c));
            }
        });
    });
    let lbufs: Vec<Vec<(u64, u64, &'static [u8], &'static [u8])>> = lbufs.into_iter().map(|c| c.into_inner()).collect();
    let npairs: usize = lbufs.iter().map(|b| b.len()).sum();
    let mut idx = HMap::new(npairs * 2);
    let mut ids: Vec<(u64, &'static [u8])> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(npairs);
    for b in &lbufs {
        for &(ha, hb, a, c) in b {
            let n = ids.len() as u64;
            let ia = idx.insert(ha, n);
            if ia == n {
                ids.push((ha, a));
                parent.push(n as u32);
            }
            let n = ids.len() as u64;
            let ib = idx.insert(hb, n);
            if ib == n {
                ids.push((hb, c));
                parent.push(n as u32);
            }
            edges.push((ia as u32, ib as u32));
        }
    }
    drop(idx);
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for &(a, b) in &edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    let nid = ids.len();
    let mut best: Vec<u32> = (0..nid as u32).collect();
    for i in 0..nid {
        let r = find(&mut parent, i as u32) as usize;
        if ids[i].1 < ids[best[r] as usize].1 {
            best[r] = i as u32;
        }
    }
    let mut person_map = HMap::new(nid);
    let mut linked_names = Table2::new(nid);
    for i in 0..nid {
        let r = find(&mut parent, i as u32) as usize;
        let (hn, name) = ids[best[r] as usize];
        person_map.set(ids[i].0, hn);
        linked_names.insert(hn, name);
    }
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
        linked_names,
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

    type EBufs = Vec<Vec<ERec>>;
    type CBufs = Vec<Vec<CRec>>;
    let ebufs: Vec<SyncCell<EBufs>> = (0..nt).map(|_| SyncCell::new((0..NPART).map(|_| Vec::new()).collect())).collect();
    let cbufs: Vec<SyncCell<CBufs>> = (0..nt).map(|_| SyncCell::new((0..NPART).map(|_| Vec::new()).collect())).collect();
    par_for(units.len(), nt, |t, u| {
        let (is_click, fi, s, e) = units[u];
        if !is_click {
            let eb = unsafe { ebufs[t].get() };
            let d = ev_files[fi];
            const B: usize = 16;
            let mut batch: [ERec; B] = unsafe { std::mem::zeroed() };
            let mut nb = 0usize;
            let link = &ctx.link;
            let flush = |batch: &mut [ERec; B], nb: usize, eb: &mut EBufs| {
                for r in batch[..nb].iter_mut() {
                    r.person = link.get(r.person).unwrap_or(r.person);
                    eb[(r.eid >> 56) as usize].push(*r);
                }
            };
            for_rows::<7>(d, s, e, |[eid, uid, ts, ing, typ, pid, price]| {
                if eid.is_empty() {
                    return;
                }
                let ingv = parse_ts(ing).unwrap_or(i64::MIN);
                let mut r = ERec {
                    eid: hash(eid),
                    ing: ingv,
                    person: 0,
                    uidp: 0,
                    eid2: hash_seed(eid, 0x1357_9BDF_2468_ACE0) as u32,
                    ts: 0,
                    price: 0,
                    kind: 0,
                    cat: 0,
                };
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
                            r.price = p;
                            r.cat = ctx.products.get(pid).map(|c| c as u8).unwrap_or(ctx.unknown_cat);
                        }
                        None => ok = false,
                    }
                }
                if ok {
                    r.kind = kind;
                    let h = hash(uid);
                    link.prefetch(h);
                    r.person = h;
                    r.uidp = uid.as_ptr() as u64;
                    batch[nb] = r;
                    nb += 1;
                    if nb == B {
                        flush(&mut batch, nb, eb);
                        nb = 0;
                    }
                } else {
                    eb[(r.eid >> 56) as usize].push(r);
                }
            });
            flush(&mut batch, nb, eb);
        } else {
            let cb = unsafe { cbufs[t].get() };
            let d = ck_files[fi];
            const B: usize = 16;
            let mut batch: [CRec; B] = unsafe { std::mem::zeroed() };
            let mut nb = 0usize;
            let link = &ctx.link;
            let flush = |batch: &mut [CRec; B], nb: usize, cb: &mut CBufs| {
                for r in batch[..nb].iter_mut() {
                    r.person = link.get(r.person).unwrap_or(r.person);
                    cb[(r.person >> 56) as usize].push(*r);
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
                    flush(&mut batch, nb, cb);
                    nb = 0;
                }
            });
            flush(&mut batch, nb, cb);
        }
    });
    let ebufs: Vec<EBufs> = ebufs.into_iter().map(|c| c.into_inner()).collect();
    let cbufs: Vec<CBufs> = cbufs.into_iter().map(|c| c.into_inner()).collect();
    lap("parsed");

    // ---------------------------------------------------------------- dedup events
    let sbufs: Vec<SyncCell<Vec<Vec<SRec>>>> =
        (0..nt).map(|_| SyncCell::new((0..NPART).map(|_| Vec::new()).collect())).collect();
    par_for(NPART, nt, |t, p| {
        let n: usize = ebufs.iter().map(|b| b[p].len()).sum();
        let mut v: Vec<ERec> = Vec::with_capacity(n);
        for b in &ebufs {
            v.extend_from_slice(&b[p]);
        }
        let cap = (v.len() * 2 + 16).next_power_of_two();
        let mask = cap - 1;
        // slot: (eid, eid2, index+1)
        let mut tab: Vec<(u64, u32, u32)> = vec![(0, 0, 0); cap];
        for (ri, r) in v.iter().enumerate() {
            let mut h = (r.eid as usize ^ (r.eid >> 32) as usize) & mask;
            loop {
                let sl = &mut tab[h];
                if sl.2 == 0 {
                    *sl = (r.eid, r.eid2, ri as u32 + 1);
                    break;
                }
                if sl.0 == r.eid && sl.1 == r.eid2 {
                    if r.ing < v[sl.2 as usize - 1].ing {
                        sl.2 = ri as u32 + 1;
                    }
                    break;
                }
                h = (h + 1) & mask;
            }
        }
        let sb = unsafe { sbufs[t].get() };
        for sl in &tab {
            if sl.2 == 0 {
                continue;
            }
            let r = &v[sl.2 as usize - 1];
            if r.kind != 0 {
                sb[(r.person >> 56) as usize].push(SRec {
                    person: r.person,
                    uidp: r.uidp,
                    ts: r.ts,
                    price: r.price,
                    kind: r.kind,
                    cat: r.cat,
                });
            }
        }
    });
    drop(ebufs);
    let sbufs: Vec<Vec<Vec<SRec>>> = sbufs.into_iter().map(|c| c.into_inner()).collect();
    lap("dedup");

    // ---------------------------------------------------------------- sessions, credit, assists
    let ncat = ctx.cats.len();
    let organic = ncamp;
    let urows: Vec<SyncCell<Vec<URow>>> = (0..nt).map(|_| SyncCell::new(Vec::new())).collect();
    let cagg: Vec<SyncCell<Vec<[i64; 4]>>> =
        (0..nt).map(|_| SyncCell::new(vec![[0i64; 4]; (ncamp + 1) * ctx.ndays])).collect();
    let scratch: Vec<SyncCell<(Vec<u32>, Vec<u32>, Vec<u16>)>> =
        (0..nt).map(|_| SyncCell::new((vec![0; ncamp + 1], vec![0; ncamp + 1], Vec::new()))).collect();
    par_for(NPART, nt, |t, p| {
        let n: usize = sbufs.iter().map(|b| b[p].len()).sum();
        let mut ev: Vec<SRec> = Vec::with_capacity(n);
        for b in &sbufs {
            ev.extend_from_slice(&b[p]);
        }
        ev.sort_unstable_by_key(|r| (r.person, r.ts));
        let n: usize = cbufs.iter().map(|b| b[p].len()).sum();
        let mut ck: Vec<CRec> = Vec::with_capacity(n);
        for b in &cbufs {
            ck.extend_from_slice(&b[p]);
        }
        ck.sort_unstable_by(|a, b| {
            (a.person, a.ts).cmp(&(b.person, b.ts)).then_with(|| val_at(a.cidp).cmp(val_at(b.cidp)))
        });
        let rows = unsafe { urows[t].get() };
        let ca = unsafe { cagg[t].get() };
        let (cnt, pos, active) = unsafe { scratch[t].get() };
        let mut catsum = vec![0i64; ncat];
        let mut i = 0;
        let mut ci = 0;
        while i < ev.len() {
            let person = ev[i].person;
            let mut j = i + 1;
            while j < ev.len() && ev[j].person == person {
                j += 1;
            }
            while ci < ck.len() && ck[ci].person < person {
                ci += 1;
            }
            let cs = ci;
            while ci < ck.len() && ck[ci].person == person {
                ci += 1;
            }
            let cl = &ck[cs..ci];
            let es = &ev[i..j];
            let name = ctx.linked_names.get(person).unwrap_or_else(|| val_at(es[0].uidp));

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
                    if r.kind == 3 {
                        rev += r.price as i64;
                        has_p = true;
                        catsum[r.cat as usize] += r.price as i64;
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
            i = j;
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
