// Minimal Parquet writer: one row group, PLAIN pages, uncompressed.
use std::path::Path;

use crate::URow;

pub struct CRow {
    pub campaign: &'static [u8],
    pub channel: &'static [u8],
    pub day: i32,
    pub v: [i64; 4],
}

// ---------------------------------------------------------------- thrift compact
struct Tw {
    b: Vec<u8>,
    last: Vec<i16>,
}

impl Tw {
    fn new() -> Tw {
        Tw { b: Vec::new(), last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn zz(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field(&mut self, id: i16, t: u8) {
        let l = self.last.last_mut().unwrap();
        let d = id - *l;
        if d > 0 && d <= 15 {
            self.b.push(((d as u8) << 4) | t);
        } else {
            self.b.push(t);
            let z = Tw::zz(id as i64);
            self.varint(z);
        }
        *self.last.last_mut().unwrap() = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.varint(Tw::zz(v as i64));
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.varint(Tw::zz(v));
    }
    fn binf(&mut self, id: i16, v: &[u8]) {
        self.field(id, 8);
        self.varint(v.len() as u64);
        self.b.extend_from_slice(v);
    }
    fn list(&mut self, id: i16, et: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.b.push(((n as u8) << 4) | et);
        } else {
            self.b.push(0xF0 | et);
            self.varint(n as u64);
        }
    }
    fn sbegin(&mut self, id: i16) {
        self.field(id, 12);
        self.last.push(0);
    }
    fn elem_begin(&mut self) {
        self.last.push(0);
    }
    fn send(&mut self) {
        self.b.push(0);
        self.last.pop();
    }
    fn stop(&mut self) {
        self.b.push(0);
    }
}

// ---------------------------------------------------------------- columns
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Str,
    Date,
    I64,
}

struct Col {
    name: &'static str,
    kind: Kind,
    optional: bool,
    // per page: (plain values, def level bytes (optional cols), num rows)
    pages: Vec<(Vec<u8>, Vec<u8>, usize)>,
}

fn ptype(k: Kind) -> i32 {
    match k {
        Kind::Str => 6,
        Kind::Date => 1,
        Kind::I64 => 2,
    }
}

const PAGE_ROWS: usize = 1 << 20;

/// Bit-packed RLE-hybrid definition levels (bit width 1), with 4-byte length prefix.
fn def_levels(defs: &[bool]) -> Vec<u8> {
    let groups = defs.len().div_ceil(8);
    let mut body = Vec::with_capacity(groups + 10);
    let mut h = ((groups as u64) << 1) | 1;
    while h >= 0x80 {
        body.push((h as u8) | 0x80);
        h >>= 7;
    }
    body.push(h as u8);
    for g in 0..groups {
        let mut byte = 0u8;
        for bit in 0..8 {
            let i = g * 8 + bit;
            if i < defs.len() && defs[i] {
                byte |= 1 << bit;
            }
        }
        body.push(byte);
    }
    let mut out = (body.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&body);
    out
}

fn write_file(path: &Path, cols: &[Col], nrows: usize) {
    let mut f: Vec<u8> = b"PAR1".to_vec();
    // (offset, total bytes, num values)
    let mut meta: Vec<(i64, i64, i64)> = Vec::new();
    if nrows > 0 {
        for c in cols {
            let off = f.len() as i64;
            for (vals, defs, n) in &c.pages {
                let size = (vals.len() + defs.len()) as i32;
                let mut h = Tw::new();
                h.i32f(1, 0); // DATA_PAGE
                h.i32f(2, size);
                h.i32f(3, size);
                h.sbegin(5);
                h.i32f(1, *n as i32);
                h.i32f(2, 0); // PLAIN
                h.i32f(3, 3); // RLE
                h.i32f(4, 3); // RLE
                h.send();
                h.stop();
                f.extend_from_slice(&h.b);
                f.extend_from_slice(defs);
                f.extend_from_slice(vals);
            }
            meta.push((off, f.len() as i64 - off, nrows as i64));
        }
    }
    let mut m = Tw::new();
    m.i32f(1, 1);
    m.list(2, 12, cols.len() + 1);
    m.elem_begin();
    m.binf(4, b"schema");
    m.i32f(5, cols.len() as i32);
    m.send();
    for c in cols {
        m.elem_begin();
        m.i32f(1, ptype(c.kind));
        m.i32f(3, if c.optional { 1 } else { 0 });
        m.binf(4, c.name.as_bytes());
        match c.kind {
            Kind::Str => {
                m.i32f(6, 0); // UTF8
                m.sbegin(10);
                m.sbegin(1);
                m.send();
                m.send();
            }
            Kind::Date => {
                m.i32f(6, 6); // DATE
                m.sbegin(10);
                m.sbegin(6);
                m.send();
                m.send();
            }
            Kind::I64 => {}
        }
        m.send();
    }
    m.i64f(3, nrows as i64);
    if nrows > 0 {
        m.list(4, 12, 1);
        m.elem_begin();
        m.list(1, 12, cols.len());
        let mut total = 0i64;
        for (c, &(off, sz, nv)) in cols.iter().zip(meta.iter()) {
            total += sz;
            m.elem_begin();
            m.i64f(2, off);
            m.sbegin(3);
            m.i32f(1, ptype(c.kind));
            m.list(2, 5, 2);
            m.varint(Tw::zz(0));
            m.varint(Tw::zz(3));
            m.list(3, 8, 1);
            m.varint(c.name.len() as u64);
            m.b.extend_from_slice(c.name.as_bytes());
            m.i32f(4, 0); // UNCOMPRESSED
            m.i64f(5, nv);
            m.i64f(6, sz);
            m.i64f(7, sz);
            m.i64f(9, off);
            m.send();
            m.send();
        }
        m.i64f(2, total);
        m.i64f(3, nrows as i64);
        m.send();
    } else {
        m.list(4, 12, 0);
    }
    m.binf(6, b"hcmut-tmt sessions");
    m.stop();
    f.extend_from_slice(&m.b);
    f.extend_from_slice(&(m.b.len() as u32).to_le_bytes());
    f.extend_from_slice(b"PAR1");
    std::fs::write(path, f).expect("write parquet");
}

fn str_col(name: &'static str, n: usize, get: impl Fn(usize) -> Option<&'static [u8]>, optional: bool) -> Col {
    let mut pages = Vec::new();
    let mut s = 0;
    while s < n {
        let e = (s + PAGE_ROWS).min(n);
        let mut vals = Vec::new();
        let mut defs = Vec::new();
        for i in s..e {
            match get(i) {
                Some(v) => {
                    vals.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    vals.extend_from_slice(v);
                    defs.push(true);
                }
                None => defs.push(false),
            }
        }
        let d = if optional { def_levels(&defs) } else { Vec::new() };
        pages.push((vals, d, e - s));
        s = e;
    }
    Col { name, kind: Kind::Str, optional, pages }
}

fn int_col(name: &'static str, kind: Kind, n: usize, get: impl Fn(usize) -> i64) -> Col {
    let mut pages = Vec::new();
    let mut s = 0;
    while s < n {
        let e = (s + PAGE_ROWS).min(n);
        let mut vals = Vec::with_capacity((e - s) * 8);
        for i in s..e {
            if kind == Kind::Date {
                vals.extend_from_slice(&(get(i) as i32).to_le_bytes());
            } else {
                vals.extend_from_slice(&get(i).to_le_bytes());
            }
        }
        pages.push((vals, Vec::new(), e - s));
        s = e;
    }
    Col { name, kind, optional: false, pages }
}

pub fn write_user_daily(path: &Path, rows: &[URow], cats: &[&'static [u8]]) {
    let n = rows.len();
    let cols = vec![
        str_col("user_id", n, |i| Some(rows[i].name), false),
        int_col("day", Kind::Date, n, |i| rows[i].day as i64),
        int_col("sessions", Kind::I64, n, |i| rows[i].sessions),
        int_col("events", Kind::I64, n, |i| rows[i].events),
        int_col("revenue_cents", Kind::I64, n, |i| rows[i].rev),
        str_col("top_category", n, |i| if rows[i].top < 0 { None } else { Some(cats[rows[i].top as usize]) }, true),
    ];
    write_file(path, &cols, n);
}

pub fn write_campaign_daily(path: &Path, rows: &[CRow]) {
    let n = rows.len();
    let cols = vec![
        str_col("campaign_id", n, |i| Some(rows[i].campaign), false),
        str_col("channel", n, |i| Some(rows[i].channel), false),
        int_col("day", Kind::Date, n, |i| rows[i].day as i64),
        int_col("sessions", Kind::I64, n, |i| rows[i].v[0]),
        int_col("purchasing_sessions", Kind::I64, n, |i| rows[i].v[1]),
        int_col("revenue_cents", Kind::I64, n, |i| rows[i].v[2]),
        int_col("assisted_sessions", Kind::I64, n, |i| rows[i].v[3]),
    ];
    write_file(path, &cols, n);
}
