use crate::vm::Value;

#[derive(Debug, PartialEq)]
pub enum VMError {
    InvalidValueError(u16),
    InvalidOpcode(Value),
    InvalidProgramCounter(usize),
    InvalidMemoryAddress(usize),
}
