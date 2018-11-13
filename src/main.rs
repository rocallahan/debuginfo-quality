#[macro_use]
extern crate clap;
extern crate cpp_demangle;
#[macro_use]
extern crate derive_more;
extern crate fallible_iterator;
extern crate gimli;
extern crate memmap;
extern crate object;
extern crate rayon;
extern crate regex;
extern crate structopt;
extern crate typed_arena;

use std::borrow::{Borrow, Cow};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::error;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::iter::Iterator;
use std::path::{Path, PathBuf};
use std::process;

use cpp_demangle::*;
use fallible_iterator::FallibleIterator;
use gimli::{AttributeValue, CompilationUnitHeader, EndianSlice};
use object::Object;
use rayon::prelude::*;
use regex::Regex;
use structopt::StructOpt;
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

arg_enum! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    enum Language {
        Cpp,
        Rust,
    }
}

#[derive(StructOpt, Clone)]
/// Evaluate the quality of debuginfo
#[structopt(name = "debuginfo-quality")]
struct Opt {
    /// Show results for each function. Print the worst functions first.
    #[structopt(short = "f", long = "functions")]
    functions: bool,
    /// Show results for each variable. Print the worst functions first.
    #[structopt(short = "v", long = "variables")]
    variables: bool,
    /// Regex to match function names against
    #[structopt(short = "s", long="select-functions")]
    select_functions: Option<Regex>,
    /// Languages to look at
    #[structopt(short = "l", long="language", raw(possible_values = "&Language::variants()", case_insensitive = "true"))]
    language: Option<Language>,
    /// File to analyze
    #[structopt(parse(from_os_str))]
    file: PathBuf,
    /// File to use as a baseline. We try to match up functions in this file
    /// against functions in the main file; for all matches, subtract the scope coverage
    /// percentage in the baseline from the percentage of the main file.
    #[structopt(parse(from_os_str))]
    baseline: Option<PathBuf>,
}

fn map(path: &Path) -> memmap::Mmap {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!(
                "Failed to open file '{}': {}",
                path.display(),
                error::Error::description(&err)
            );
            process::exit(1);
        }
    };
    match unsafe { memmap::Mmap::map(&file) } {
        Ok(mmap) => mmap,
        Err(err) => {
            eprintln!("Failed to map file '{}': {}", path.display(), &err);
            process::exit(1);
        }
    }
}

fn open<'a>(path: &Path, mmap: &'a memmap::Mmap) -> object::File<'a> {
    let file = match object::File::parse(&*mmap) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to parse file '{}': {}", path.display(), err);
            process::exit(1);
        }
    };
    assert!(file.is_little_endian());
    file
}

fn write_stats<W: io::Write>(mut w: W, stats: &VariableStats, base_stats: Option<&VariableStats>) {
    if let Some(b) = base_stats {
        writeln!(w, "\t{}\t{}\t{}\t{}\t{}\t{}\t{}", stats.instruction_bytes_defined,
                 stats.instruction_bytes_in_scope,
                 stats.fraction_defined(), b.instruction_bytes_defined,
                 b.instruction_bytes_in_scope,
                 b.fraction_defined(), stats.fraction_defined() - b.fraction_defined()).unwrap();
    } else {
        writeln!(w, "\t{}\t{}\t{}", stats.instruction_bytes_defined,
                 stats.instruction_bytes_in_scope,
                 stats.fraction_defined()).unwrap();
    }
}

fn write_stats_label<W: io::Write>(mut w: W, label: &str, stats: &VariableStats, base_stats: Option<&VariableStats>) {
    write!(w, "{}", label).unwrap();
    write_stats(w, stats, base_stats);
}

fn main() {
    let opt = Opt::from_args();
    let file_map = map(&opt.file);
    let file = open(&opt.file, &file_map);
    let baseline_map = opt.baseline.as_ref().map(|p| (p, map(p)));
    let baseline_file = baseline_map.as_ref().map(|&(ref p, ref m)| open(p, m));

    fn load_section<'a, 'file, 'input, S>(
        arena: &'a Arena<Cow<'file, [u8]>>,
        file: &'file object::File<'input>,
    ) -> S
    where
        S: gimli::Section<gimli::EndianSlice<'a, gimli::LittleEndian>>,
        'file: 'input,
        'a: 'file
    {
        let data = file.section_data_by_name(S::section_name()).unwrap_or(Cow::Borrowed(&[]));
        let data_ref = (*arena.alloc(data)).borrow();
        S::from(gimli::EndianSlice::new(data_ref, gimli::LittleEndian))
    }

    let mut stats = Stats { bundle: StatsBundle::default(), opt: opt.clone(), output: Vec::new() };
    let mut base_stats = None;

    {
        let file = &file;
        let arena = Arena::new();
        // Variables representing sections of the file. The type of each is inferred from its use in the
        // validate_info function below.
        let debug_abbrev = &load_section(&arena, file);
        let debug_info = &load_section(&arena, file);
        let debug_ranges = load_section(&arena, file);
        let debug_rnglists = load_section(&arena, file);
        let rnglists = &gimli::RangeLists::new(debug_ranges, debug_rnglists).unwrap();
        let debug_str = &load_section(&arena, file);

        let debug_loc = load_section(&arena, file);
        let debug_loclists = load_section(&arena, file);
        let loclists = &gimli::LocationLists::new(debug_loc, debug_loclists).unwrap();

        evaluate_info(debug_info, debug_abbrev, debug_str, rnglists, loclists, &mut stats);
    }

    if let Some(file) = baseline_file.as_ref() {
        let arena = Arena::new();
        // Variables representing sections of the file. The type of each is inferred from its use in the
        // validate_info function below.
        let debug_abbrev = &load_section(&arena, file);
        let debug_info = &load_section(&arena, file);
        let debug_ranges = load_section(&arena, file);
        let debug_rnglists = load_section(&arena, file);
        let rnglists = &gimli::RangeLists::new(debug_ranges, debug_rnglists).unwrap();
        let debug_str = &load_section(&arena, file);

        let debug_loc = load_section(&arena, file);
        let debug_loclists = load_section(&arena, file);
        let loclists = &gimli::LocationLists::new(debug_loc, debug_loclists).unwrap();

        let mut stats = Stats { bundle: StatsBundle::default(), opt: opt.clone(), output: Vec::new() };
        evaluate_info(debug_info, debug_abbrev, debug_str, rnglists, loclists, &mut stats);
        base_stats = Some(stats);
    }

    let stdout = io::stdout();
    let mut stdout_locked = stdout.lock();
    let mut w = BufWriter::new(&mut stdout_locked);

    if base_stats.is_some() {
        writeln!(&mut w, "\tDef\tScope\tFraction\tBaseDef\tBaseScope\tBaseFraction\tFinal").unwrap();
    } else {
        writeln!(&mut w, "\tDef\tScope\tFraction").unwrap();
    }
    writeln!(&mut w).unwrap();
    if stats.opt.functions || stats.opt.variables {
        let mut functions: Vec<(FunctionStats, Option<&FunctionStats>)> = if let Some(base) = base_stats.as_ref() {
            let mut base_functions = HashMap::new();
            for f in base.output.iter() {
                base_functions.insert(&f.name, f);
            }
            stats.output.into_iter().filter_map(|o| base_functions.get(&o.name).map(|b| (o, Some(*b)))).collect()
        } else {
            stats.output.into_iter().map(|o| (o, None)).collect()
        };
        functions.sort_by(|a, b| goodness(a).partial_cmp(&goodness(b)).unwrap());
        for (function_stats, base_function_stats) in functions {
            if stats.opt.variables {
                for v in function_stats.variables {
                    write!(&mut w, "{}", &function_stats.name);
                    for inline in v.inlines {
                        write!(&mut w, ",{}", &inline);
                    }
                    write!(&mut w, ",{}@0x{:x}:0x{:x}", &v.name, function_stats.unit_offset, v.entry_offset);
                    write_stats(&mut w, &v.stats, None);
                }
            } else {
                write!(&mut w, "{}@0x{:x}:0x{:x}", &function_stats.name, function_stats.unit_offset, function_stats.entry_offset);
                write_stats(&mut w, &function_stats.stats, base_function_stats.map(|b| &b.stats));
            }
        }
        writeln!(&mut w).unwrap();
    }
    write_stats_label(&mut w, "params", &stats.bundle.parameters, base_stats.as_ref().map(|b| &b.bundle.parameters));
    write_stats_label(&mut w, "vars", &stats.bundle.variables, base_stats.as_ref().map(|b| &b.bundle.variables));
    let all = stats.bundle.variables + stats.bundle.parameters;
    let base_all = base_stats.as_ref().map(|b| b.bundle.variables.clone() + b.bundle.parameters.clone());
    write_stats_label(&mut w, "all", &all, base_all.as_ref());
}

fn goodness(&(ref a, ref a_base): &(FunctionStats, Option<&FunctionStats>)) -> (f64, i64) {
    (if let Some(a_base) = a_base.as_ref() {
        a.stats.fraction_defined() - a_base.stats.fraction_defined()
    } else {
        a.stats.fraction_defined()
    }, -(a.stats.instruction_bytes_in_scope as i64))
}

#[derive(Clone, Default, Add, AddAssign)]
struct StatsBundle {
    parameters: VariableStats,
    variables: VariableStats,
}

#[derive(Clone)]
struct Stats {
    bundle: StatsBundle,
    opt: Opt,
    output: Vec<FunctionStats>,
}

#[derive(Clone)]
struct NamedVarStats {
    inlines: Vec<String>,
    name: String,
    entry_offset: usize,
    stats: VariableStats,
}

#[derive(Clone)]
struct FunctionStats {
    name: String,
    unit_offset: usize,
    entry_offset: usize,
    stats: VariableStats,
    variables: Vec<NamedVarStats>,
}

struct UnitStats<'a> {
    bundle: StatsBundle,
    opt: &'a Opt,
    noninline_function_stack: Vec<Option<FunctionStats>>,
    output: Vec<FunctionStats>,
}

struct FinalUnitStats {
    bundle: StatsBundle,
    output: Vec<FunctionStats>,
}

impl<'a> From<UnitStats<'a>> for FinalUnitStats {
    fn from(v: UnitStats<'a>) -> Self {
        FinalUnitStats {
            bundle: v.bundle,
            output: v.output,
        }
    }
}

impl Stats {
    fn new_unit_stats(&self) -> UnitStats {
        UnitStats {
            bundle: StatsBundle::default(),
            opt: &self.opt,
            noninline_function_stack: Vec::new(),
            output: Vec::new(),
        }
    }
    fn accumulate(&mut self, mut stats: FinalUnitStats) {
        self.bundle += stats.bundle;
        self.output.append(&mut stats.output);
    }
}

impl<'a> UnitStats<'a> {
    fn enter_noninline_function(&mut self, name: &MaybeDemangle<'a>, unit_offset: usize, entry_offset: usize) {
        let demangled = name.demangled();
        self.noninline_function_stack.push(if self.opt.select_functions.as_ref().map(|r| r.is_match(&demangled)).unwrap_or(true) {
            Some(FunctionStats {
                name: demangled.into_owned(),
                unit_offset: unit_offset,
                entry_offset: entry_offset,
                stats: VariableStats::default(),
                variables: Vec::new(),
            })
        } else {
            None
        });
    }
    fn accumulate(&mut self,
                  var_type: VarType,
                  entry_offset: usize,
                  subprogram_name_stack: &[(MaybeDemangle, isize, bool)],
                  var_name: Option<MaybeDemangle>,
                  stats: VariableStats) {
        let function_stats = if let Some(s) = self.noninline_function_stack.last_mut().unwrap().as_mut() {
            s
        } else {
            return;
        };
        let mut i = subprogram_name_stack.len();
        while i > 0 && subprogram_name_stack[i - 1].2 {
            i -= 1;
        }
        function_stats.stats += stats.clone();
        match var_type {
            VarType::Parameter => self.bundle.parameters += stats.clone(),
            VarType::Variable => self.bundle.variables += stats.clone(),
        }
        if self.opt.variables {
            function_stats.variables.push(NamedVarStats {
                inlines: subprogram_name_stack[i..].iter().map(|&(ref name, _, _)| name.demangled().into_owned()).collect(),
                name: var_name.map(|d| d.demangled()).unwrap_or(Cow::Borrowed("<anon>")).into_owned(),
                entry_offset: entry_offset,
                stats: stats,
            });
        }
    }
    fn leave_noninline_function(&mut self) {
        if let Some(function_stats) = self.noninline_function_stack.pop().unwrap() {
            if function_stats.stats.instruction_bytes_in_scope > 0 &&
                (self.opt.functions || self.opt.variables) {
                self.output.push(function_stats);
            }
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
    fn fraction_defined(&self) -> f64 {
        (self.instruction_bytes_defined as f64)/(self.instruction_bytes_in_scope as f64)
    }
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

fn to_ref_str<'abbrev, 'unit, R>(unit: &'unit CompilationUnitHeader<R, R::Offset>,
                                 entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, R>) -> String
    where R: Reader {
    format!("{:x}:{:x}", unit.offset().0, entry.offset().0)
}

enum MaybeDemangle<'a> {
    Demangle(Cow<'a, str>),
    Raw(Cow<'a, str>)
}

impl<'a> MaybeDemangle<'a> {
    fn demangled(&self) -> Cow<'a, str> {
        match self {
            &MaybeDemangle::Demangle(ref s) => {
                if let Ok(sym) = BorrowedSymbol::new(s.as_bytes()) {
                    match sym.demangle(&DemangleOptions::default()) {
                        Ok(d) => d.into(),
                        Err(_) => s.clone(),
                    }
                } else {
                    s.clone()
                }
            }
            &MaybeDemangle::Raw(ref s) => {
                s.clone()
            }
        }
    }
}

fn lookup_name<'abbrev, 'unit, 'a>(unit: &'unit CompilationUnitHeader<EndianSlice<'a, gimli::LittleEndian>, usize>,
                                   entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, EndianSlice<'a, gimli::LittleEndian>>,
                                   abbrevs: &gimli::Abbreviations,
                                   debug_str: &'a gimli::DebugStr<EndianSlice<'a, gimli::LittleEndian>>) -> Option<MaybeDemangle<'a>>
    where 'a: 'unit {
    let mut entry = entry.clone();
    loop {
        match entry.attr_value(gimli::DW_AT_linkage_name).unwrap() {
            Some(gimli::AttributeValue::String(string)) => return Some(MaybeDemangle::Demangle(string.to_string_lossy())),
            Some(gimli::AttributeValue::DebugStrRef(offset)) => return Some(MaybeDemangle::Demangle(debug_str.get_str(offset).unwrap().to_string_lossy())),
            Some(_) => panic!("Invalid DW_AT_name"),
            None => ()
        }
        match entry.attr_value(gimli::DW_AT_name).unwrap() {
            Some(gimli::AttributeValue::String(string)) => return Some(MaybeDemangle::Raw(string.to_string_lossy())),
            Some(gimli::AttributeValue::DebugStrRef(offset)) => return Some(MaybeDemangle::Raw(debug_str.get_str(offset).unwrap().to_string_lossy())),
            Some(_) => panic!("Invalid DW_AT_name"),
            None => ()
        }
        let reference = if let Some(r) = entry.attr_value(gimli::DW_AT_abstract_origin).unwrap() {
            r
        } else if let Some(r) = entry.attr_value(gimli::DW_AT_specification).unwrap() {
            r
        } else {
            return None;
        };
        match reference {
            gimli::AttributeValue::UnitRef(offset) => {
                entry = unit.entries_at_offset(abbrevs, offset).unwrap().next_dfs().unwrap().unwrap().1.clone();
            },
            _ => {
                panic!("Unexpected attribute value for reference: {:?}", reference);
            }
        }
    }
}

fn evaluate_info<'a>(
    debug_info: &'a gimli::DebugInfo<EndianSlice<'a, gimli::LittleEndian>>,
    debug_abbrev: &'a gimli::DebugAbbrev<EndianSlice<'a, gimli::LittleEndian>>,
    debug_str: &'a gimli::DebugStr<EndianSlice<'a, gimli::LittleEndian>>,
    rnglists: &gimli::RangeLists<EndianSlice<'a, gimli::LittleEndian>>,
    loclists: &gimli::LocationLists<EndianSlice<'a, gimli::LittleEndian>>,
    stats: &'a mut Stats
)
{
    let units = debug_info.units().collect::<Vec<_>>().unwrap();
    let process_unit = |stats: &Stats, unit: CompilationUnitHeader<EndianSlice<'a, gimli::LittleEndian>, usize>| -> FinalUnitStats {
        let mut unit_stats = stats.new_unit_stats();
        let abbrevs = unit.abbreviations(debug_abbrev).unwrap();
        let mut entries = unit.entries(&abbrevs);
        let mut base_address = None;
        {
            let (delta, entry) = entries.next_dfs().unwrap().unwrap();
            assert_eq!(delta, 0);
            if let Some(gimli::AttributeValue::Addr(addr)) = entry.attr_value(gimli::DW_AT_low_pc).unwrap() {
                base_address = Some(addr);
            }
            let producer = match entry.attr_value(gimli::DW_AT_producer).unwrap() {
                Some(gimli::AttributeValue::String(string)) => string.to_string_lossy(),
                Some(gimli::AttributeValue::DebugStrRef(offset)) => debug_str.get_str(offset).unwrap().to_string_lossy(),
                Some(_) => panic!("Invalid DW_AT_producer"),
                None => Cow::Borrowed(""),
            };
            let language = if producer.contains("rustc version") {
                Language::Rust
            } else {
                Language::Cpp
            };
            if stats.opt.language.map(|l| l != language).unwrap_or(false) {
                return unit_stats.into();
            }
        }
        let mut depth = 0;
        let mut scopes: Vec<(Vec<gimli::Range>, isize)> = Vec::new();
        let mut namespace_stack: Vec<(MaybeDemangle, isize, bool)> = Vec::new();
        loop {
            let (delta, entry) = match entries.next_dfs().unwrap() {
                None => break,
                Some(entry) => entry,
            };
            depth += delta;
            while scopes.last().map(|v| v.1 >= depth).unwrap_or(false) {
                scopes.pop();
            }
            while namespace_stack.last().map(|v| v.1 >= depth).unwrap_or(false) {
                if !namespace_stack.pop().unwrap().2 {
                    unit_stats.leave_noninline_function();
                }
            }
            if let Some(AttributeValue::RangeListsRef(offset)) = entry.attr_value(gimli::DW_AT_ranges).unwrap() {
                let rs = rnglists.ranges(offset, unit.version(), unit.address_size(), base_address.unwrap()).unwrap();
                let mut bytes_ranges = rs.collect::<Vec<_>>().unwrap();
                sort_nonoverlapping(&mut bytes_ranges);
                scopes.push((bytes_ranges, depth));
            } else if let Some(AttributeValue::Udata(data)) = entry.attr_value(gimli::DW_AT_high_pc).unwrap() {
                if let Some(gimli::AttributeValue::Addr(addr)) = entry.attr_value(gimli::DW_AT_low_pc).unwrap() {
                    let bytes_range = gimli::Range { begin: addr, end: addr + data };
                    scopes.push((vec![bytes_range], depth));
                }
            }
            let var_type = match entry.tag() {
                gimli::DW_TAG_formal_parameter => VarType::Parameter,
                gimli::DW_TAG_variable => VarType::Variable,
                gimli::DW_TAG_subprogram => {
                    if let Some(name) = lookup_name(&unit, &entry, &abbrevs, debug_str) {
                        unit_stats.enter_noninline_function(&name, unit.offset().0, entry.offset().0);
                        namespace_stack.push((name, depth, false));
                    }
                    continue;
                }
                gimli::DW_TAG_inlined_subroutine => {
                    if let Some(name) = lookup_name(&unit, &entry, &abbrevs, debug_str) {
                        namespace_stack.push((name, depth, true));
                    }
                    continue;
                }
                _ => continue,
            };
            let ranges = if let Some(s) = scopes.last() {
                if s.1 + 1 == depth && !s.0.is_empty() {
                    &s.0[..]
                } else {
                    continue;
                }
            } else {
                continue;
            };
            let var_stats = if entry.attr_value(gimli::DW_AT_const_value).unwrap().is_some() {
                let in_scope = ranges_instruction_bytes(ranges);
                VariableStats {
                    instruction_bytes_in_scope: in_scope,
                    instruction_bytes_defined: in_scope,
                }
            } else {
                match entry.attr_value(gimli::DW_AT_location).unwrap() {
                    Some(AttributeValue::Exprloc(_)) => {
                        let in_scope = ranges_instruction_bytes(ranges);
                        VariableStats {
                            instruction_bytes_in_scope: in_scope,
                            instruction_bytes_defined: in_scope,
                        }
                    }
                    Some(AttributeValue::LocationListsRef(loc)) => {
                        let mut locations = {
                            let iter =
                              loclists.locations(loc, unit.version(), unit.address_size(), base_address.unwrap())
                                  .expect("invalid location list");
                            iter.map(|e| e.range).collect::<Vec<_>>().expect("invalid location list")
                        };
                        sort_nonoverlapping(&mut locations);
                        VariableStats {
                            instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                            instruction_bytes_defined: ranges_overlap_instruction_bytes(ranges, &locations[..]),
                        }
                    }
                    None => {
                        VariableStats {
                            instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                            instruction_bytes_defined: 0,
                        }
                    }
                    _ => panic!("Unknown DW_AT_location attribute at {}", to_ref_str(&unit, &entry)),
                }
            };
            let var_name = lookup_name(&unit, &entry, &abbrevs, debug_str);
            unit_stats.accumulate(var_type, entry.offset().0,
                                  &namespace_stack, var_name, var_stats);
        }
        unit_stats.into()
    };
    let all_stats = units.into_par_iter().map(|u| process_unit(stats, u)).collect::<Vec<_>>();
    for s in all_stats {
        stats.accumulate(s);
    }
}
