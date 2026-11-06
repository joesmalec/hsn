//! P2.9 tests of the CPU oracle: the header's constants equal `spec.rs`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use hsn_kernels::{DType, Error, spec};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every `#define HSN_<NAME> <integer>` of the header (the include guard has no value).
fn header_defines() -> BTreeMap<String, i64> {
    let path = repo_root().join("kernels/include/hsn_ops.h");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next() != Some("#define") {
                return None;
            }
            let name = words.next()?;
            let value = words.next()?.parse::<i64>().ok()?;
            name.starts_with("HSN_").then(|| (name.to_owned(), value))
        })
        .collect()
}

#[test]
fn test_p2_9_header_constants_match_spec() {
    let expected: BTreeMap<String, i64> = spec::HEADER_CONSTANTS
        .iter()
        .map(|&(name, value)| (name.to_owned(), value))
        .collect();
    assert_eq!(header_defines(), expected);

    assert_eq!(DType::F32.abi_code(), Some(spec::DTYPE_F32));
    assert_eq!(DType::F16.abi_code(), Some(spec::DTYPE_F16));
    assert_eq!(DType::BF16.abi_code(), Some(spec::DTYPE_BF16));
    assert_eq!(DType::F64.abi_code(), None);
    let statuses = [
        (
            Error::InvalidArgument(String::new()),
            spec::ERR_INVALID_ARGUMENT,
        ),
        (
            Error::UnsupportedDType {
                backend: "cpu",
                dtype: DType::F16,
            },
            spec::ERR_UNSUPPORTED_DTYPE,
        ),
        (
            Error::IndexOutOfRange(String::new()),
            spec::ERR_INDEX_OUT_OF_RANGE,
        ),
        (
            Error::DuplicateIndex(String::new()),
            spec::ERR_DUPLICATE_INDEX,
        ),
        (Error::SizeOverflow(String::new()), spec::ERR_SIZE_OVERFLOW),
        (Error::BadGraph(String::new()), spec::ERR_BAD_GRAPH),
        (Error::Backend(String::new()), spec::ERR_BACKEND),
    ];
    for (error, code) in statuses {
        assert_eq!(error.status(), code, "{error}");
        assert_ne!(code, spec::OK);
    }
}
