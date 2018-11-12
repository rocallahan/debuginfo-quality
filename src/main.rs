#[macro_use]
extern crate derive_more;
extern crate fallible_iterator;
extern crate gimli;
extern crate memmap;
extern crate object;
extern crate rayon;
extern crate structopt;
extern crate typed_arena;

use std::borrow::{Borrow, Cow};
use std::cmp::{max, min};
use std::env;
use std::fs;
use std::iter::Iterator;
use std::path::Path;
use std::error;

use fallible_iterator::FallibleIterator;
use gimli::{AttributeValue, CompilationUnitHeader};
use object::Object;
use rayon::prelude::*;
use typed_arena::Arena;

trait Reader: gimli::Reader<Offset = usize> + Send + Sync {
    type SyncSendEndian: gimli::Endianity + Send + Sync;
}

impl<'input, Endian> Reader for gimli::EndianSlice<'input, Endian>
where
    Endian: gimli::Endianity + Send + Sync,
{
    type SyncSendEndian = Endian;
}

fn main() {
    for arg in env::args_os().skip(1) {
        let path = Path::new(&arg);
        let file = match fs::File::open(&path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!(
                    "Failed to open file '{}': {}",
                    path.display(),
                    error::Error::description(&err)
                );
                continue;
            }
        };
        let file = match unsafe { memmap::Mmap::map(&file) } {
            Ok(mmap) => mmap,
            Err(err) => {
                eprintln!("Failed to map file '{}': {}", path.display(), &err);
                continue;
            }
        };
        let file = match object::File::parse(&*file) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("Failed to parse file '{}': {}", path.display(), err);
                continue;
            }
        };

        let endian = if file.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };
        evaluate_file(path, &file, endian);
    }
}

fn evaluate_file<Endian>(path: &Path, file: &object::File, endian: Endian)
where
    Endian: gimli::Endianity + Send + Sync,
{
    let arena = Arena::new();

    fn load_section<'a, 'file, 'input, S, Endian>(
        arena: &'a Arena<Cow<'file, [u8]>>,
        file: &'file object::File<'input>,
        endian: Endian,
    ) -> S
    where
        S: gimli::Section<gimli::EndianSlice<'a, Endian>>,
        Endian: gimli::Endianity + Send + Sync,
        'file: 'input,
        'a: 'file
    {
        let data = file.section_data_by_name(S::section_name()).unwrap_or(Cow::Borrowed(&[]));
        let data_ref = (*arena.alloc(data)).borrow();
        S::from(gimli::EndianSlice::new(data_ref, endian))
    }

    // Variables representing sections of the file. The type of each is inferred from its use in the
    // validate_info function below.
    let debug_abbrev = &load_section(&arena, file, endian);
    let debug_info = &load_section(&arena, file, endian);
    let debug_ranges = load_section(&arena, file, endian);
    let debug_rnglists = load_section(&arena, file, endian);
    let rnglists = &gimli::RangeLists::new(debug_ranges, debug_rnglists).unwrap();
    let debug_str = &load_section(&arena, file, endian);

    let debug_loc = load_section(&arena, file, endian);
    let debug_loclists = load_section(&arena, file, endian);
    let loclists = &gimli::LocationLists::new(debug_loc, debug_loclists).unwrap();

    let mut stats = WholeFileStats::default();
    evaluate_info(path, debug_info, debug_abbrev, debug_str, rnglists, loclists, endian, &mut stats);
    println!("\tDef\tScope");
    println!("params\t{}\t{}\t{}", stats.parameters.instruction_bytes_defined, stats.parameters.instruction_bytes_in_scope,
             stats.parameters.percent_defined());
    println!("vars\t{}\t{}\t{}", stats.variables.instruction_bytes_defined, stats.variables.instruction_bytes_in_scope,
             stats.variables.percent_defined());
    let all = stats.variables + stats.parameters;
    println!("vars\t{}\t{}\t{}", all.instruction_bytes_defined, all.instruction_bytes_in_scope,
             all.percent_defined());
}

#[derive(Default, Add, AddAssign)]
struct WholeFileStats {
    parameters: VariableStats,
    variables: VariableStats,
}

impl Stats for WholeFileStats {
    type UnitStats = WholeFileStats;
    fn new_unit_stats(&self) -> WholeFileStats { WholeFileStats::default() }
    fn accumulate(&mut self, stats: WholeFileStats) {
        *self += stats;
    }
}

impl PerUnitStats for WholeFileStats {
    fn accumulate(&mut self,
                  var_type: VarType,
                  stats: VariableStats) {
        match var_type {
            VarType::Parameter => self.parameters += stats,
            VarType::Variable => self.variables += stats,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VarType {
    Parameter,
    Variable
}

#[derive(Clone, Debug, Add, AddAssign, Default)]
struct VariableStats {
    instruction_bytes_in_scope: u64,
    instruction_bytes_defined: u64,
}

impl VariableStats {
    fn percent_defined(&self) -> f64 {
        (self.instruction_bytes_defined as f64)/(self.instruction_bytes_in_scope as f64)
    }
}

trait PerUnitStats {
    fn accumulate(&mut self,
                  var_type: VarType,
                  stats: VariableStats);
}

trait Stats {
    type UnitStats: PerUnitStats + Send;
    fn new_unit_stats(&self) -> Self::UnitStats;
    fn accumulate(&mut self, stats: Self::UnitStats);
}

fn ranges_instruction_bytes(r: &[gimli::Range]) -> u64 {
    r.iter().fold(0, |sum, r| { sum + (r.end - r.begin) })
}

fn ranges_overlap_instruction_bytes(rs1: &[gimli::Range], rs2: &[gimli::Range]) -> u64 {
    let mut iter1 = rs1.iter();
    let mut iter2 = rs2.iter();
    let mut r1_opt: Option<gimli::Range> = iter1.next().map(|r| *r);
    let mut r2_opt: Option<gimli::Range> = iter2.next().map(|r| *r);
    let mut total = 0;
    while let (Some(r1), Some(r2)) = (r1_opt, r2_opt) {
        let overlap_start = max(r1.begin, r2.begin);
        let overlap_end = min(r1.end, r2.end);
        if overlap_start < overlap_end {
            total += overlap_end - overlap_start;
        }
        let new_min = overlap_end;
        r1_opt = if r1.end <= new_min {
            iter1.next().map(|r| *r)
        } else {
            Some(gimli::Range { begin: max(r1.begin, new_min), end: r1.end })
        };
        r2_opt = if r2.end <= new_min {
            iter2.next().map(|r| *r)
        } else {
            Some(gimli::Range { begin: max(r2.begin, new_min), end: r2.end })
        };
    }
    total
}

fn sort_nonoverlapping(rs: &mut [gimli::Range]) {
    rs.sort_by_key(|r| r.begin);
    for r in 1..rs.len() {
        assert!(rs[r - 1].end <= rs[r].begin);
    }
}

fn evaluate_info<R, S>(
    path: &Path,
    debug_info: &gimli::DebugInfo<R>,
    debug_abbrev: &gimli::DebugAbbrev<R>,
    debug_str: &gimli::DebugStr<R>,
    rnglists: &gimli::RangeLists<R>,
    loclists: &gimli::LocationLists<R>,
    endian: R::SyncSendEndian,
    stats: &mut S
) where
    R: Reader,
    S: Stats + Sync + Send,
{
    let units = debug_info.units().collect::<Vec<_>>().unwrap();
    let process_unit = |stats: &S, unit: CompilationUnitHeader<R, R::Offset>| -> S::UnitStats {
        let mut unit_stats = stats.new_unit_stats();
        let abbrevs = unit.abbreviations(debug_abbrev).unwrap();
        let mut entries = unit.entries(&abbrevs);
        let mut scopes: Vec<(Vec<gimli::Range>, isize)> = Vec::new();
        let mut depth = 0;
        loop {
            let (delta, entry) = match entries.next_dfs().unwrap() {
                None => break,
                Some(entry) => entry,
            };
            depth += delta;
            while scopes.last().map(|v| v.1 >= depth).unwrap_or(false) {
                scopes.pop();
            }
            if depth == 0 {
                continue;
            }
            if let Some(AttributeValue::RangeListsRef(offset)) = entry.attr_value(gimli::DW_AT_ranges).unwrap() {
                let rs = rnglists.ranges(offset, unit.version(), unit.address_size(), 0).unwrap();
                let mut bytes_ranges = rs.collect::<Vec<_>>().unwrap();
                sort_nonoverlapping(&mut bytes_ranges);
                scopes.push((bytes_ranges, depth));
            } else if let Some(AttributeValue::Udata(data)) = entry.attr_value(gimli::DW_AT_high_pc).unwrap() {
                let bytes_range = gimli::Range { begin: 0, end: data };
                scopes.push((vec![bytes_range], depth));
            }
            let var_type = match entry.tag() {
                gimli::DW_TAG_formal_parameter => VarType::Parameter,
                gimli::DW_TAG_variable => VarType::Variable,
                _ => continue,
            };
            let relevant_scope = scopes.last().and_then(|s| if s.1 + 1 == depth { Some(&s.0[..]) } else { None });
            let var_stats = match entry.attr_value(gimli::DW_AT_location).unwrap() {
                Some(AttributeValue::Exprloc(_)) => {
                    if let Some(ranges) = relevant_scope {
                        let in_scope = ranges_instruction_bytes(ranges);
                        VariableStats {
                            instruction_bytes_in_scope: in_scope,
                            instruction_bytes_defined: in_scope,
                        }
                    } else {
                        panic!("No definition scope found for {:?}:{:?}", unit.offset(), entry.offset());
                    }
                }
                Some(AttributeValue::LocationListsRef(loc)) => {
                    if let Some(ranges) = relevant_scope {
                        let mut locations = {
                            let iter =
                              loclists.locations(loc, unit.version(), unit.address_size(), 0)
                                  .expect("invalid location list");
                            iter.map(|e| e.range).collect::<Vec<_>>().expect("invalid location list")
                        };
                        sort_nonoverlapping(&mut locations);
                        VariableStats {
                            instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                            instruction_bytes_defined: ranges_overlap_instruction_bytes(ranges, &locations[..]),
                        }
                    } else {
                        panic!("No definition scope found for {:?}:{:?}", unit.offset(), entry.offset());
                    }
                }
                None => {
                    if let Some(ranges) = relevant_scope {
                        VariableStats {
                            instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                            instruction_bytes_defined: 0,
                        }
                    } else {
                        // parameter/variable in non-definition of function
                        continue;
                    }
                }
                _ => panic!("Unknown DW_AT_location attribute at {:?}:{:?}", unit.offset(), entry.offset()),
            };
            unit_stats.accumulate(var_type, var_stats);
        }
        unit_stats
    };
    let all_stats = units.into_par_iter().map(|u| process_unit(stats, u)).collect::<Vec<_>>();
    for s in all_stats {
        stats.accumulate(s);
    }
}
