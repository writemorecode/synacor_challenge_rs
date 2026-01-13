use std::collections::VecDeque;
use std::fmt::Display;
use std::io::{self, Write};

use crate::error::VMError;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Value {
    Literal(u16),
    Register(u16),
}

impl TryFrom<u16> for Value {
    type Error = VMError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0_u16..=32767 => Ok(Value::Literal(value)),
            32768_u16..=32775 => Ok(Value::Register(value - 32768)),
            32776..=u16::MAX => Err(VMError::InvalidValueError(value)),
        }
    }
}

impl From<Value> for u16 {
    fn from(value: Value) -> Self {
        match value {
            Value::Literal(l) => l,
            Value::Register(r) => r + 32768,
        }
    }
}
impl From<&Value> for u16 {
    fn from(value: &Value) -> Self {
        match value {
            Value::Literal(l) => *l,
            Value::Register(r) => r + 32768,
        }
    }
}

const MEMORY_SIZE: usize = 2 << (16 - 1);
const REGISTER_COUNT: usize = 8;

pub struct VM {
    registers: [u16; REGISTER_COUNT],
    memory: [u16; MEMORY_SIZE],
    stack: Vec<u16>,
    pc: usize,
    input_buffer: VecDeque<u16>,
}

impl Default for VM {
    fn default() -> Self {
        Self {
            registers: [0u16; REGISTER_COUNT],
            memory: [0u16; MEMORY_SIZE],
            stack: Vec::new(),
            pc: 0,
            input_buffer: VecDeque::new(),
        }
    }
}

impl VM {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_memory(memory: [u16; MEMORY_SIZE]) -> Self {
        let mut vm = VM::new();
        vm.memory = memory;
        vm
    }

    pub fn new_with_memory_slice(mem_slice: &[u16]) -> Self {
        assert!(mem_slice.len() <= MEMORY_SIZE, "Memory slice too large.");
        let mem = {
            let mut full_mem = [0u16; MEMORY_SIZE];
            full_mem[..mem_slice.len()].copy_from_slice(mem_slice);
            full_mem
        };
        Self::new_with_memory(mem)
    }

    pub fn new_with_program_bytes(bytes: &[u8]) -> Result<Self, VMError> {
        let words = Self::decode_program_bytes(bytes)?;
        Ok(Self::new_with_memory_slice(&words))
    }

    fn decode_program_bytes(bytes: &[u8]) -> Result<Vec<u16>, VMError> {
        if !bytes.len().is_multiple_of(2) {
            return Err(VMError::InvalidBinaryLength(bytes.len()));
        }
        let mut words = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            words.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        if words.len() > MEMORY_SIZE {
            return Err(VMError::ProgramTooLarge(words.len()));
        }
        Ok(words)
    }

    pub fn read_pc(&self) -> Result<u16, VMError> {
        match self.memory.get(self.pc) {
            Some(&value) => Ok(value),
            None => Err(VMError::InvalidProgramCounter(self.pc)),
        }
    }

    pub fn read_value_at_pc(&self) -> Result<Value, VMError> {
        let instr = self.read_pc()?;
        let value = Value::try_from(instr)?;
        Ok(value)
    }

    fn decode_value(&self, address: usize) -> Result<Value, VMError> {
        let Some(&data) = self.memory.get(address) else {
            return Err(VMError::InvalidMemoryAddress(address));
        };
        Value::try_from(data)
    }

    pub fn decode_op(&mut self) -> Result<Op, VMError> {
        let value = self.read_value_at_pc()?;
        let Value::Literal(lit_val) = value else {
            return Err(VMError::InvalidOpcode(value));
        };
        let op = match lit_val {
            opcodes::OPCODE_HALT => Op::Halt,
            opcodes::OPCODE_SET => {
                let reg = self.decode_value(self.pc + 1)?;
                let val = self.decode_value(self.pc + 2)?;
                Op::Set(reg, val)
            }
            opcodes::OPCODE_PUSH => {
                let arg_value = self.decode_value(self.pc + 1)?;
                Op::Push(arg_value)
            }
            opcodes::OPCODE_POP => {
                let arg_value = self.decode_value(self.pc + 1)?;
                Op::Pop(arg_value)
            }
            opcodes::OPCODE_EQ => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Eq(d, l, r)
            }
            opcodes::OPCODE_GT => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Gt(d, l, r)
            }
            opcodes::OPCODE_JMP => {
                let dest = self.decode_value(self.pc + 1)?;
                Op::Jmp(dest)
            }
            opcodes::OPCODE_JT => {
                let cond = self.decode_value(self.pc + 1)?;
                let dest = self.decode_value(self.pc + 2)?;
                Op::Jt(cond, dest)
            }
            opcodes::OPCODE_JF => {
                let cond = self.decode_value(self.pc + 1)?;
                let dest = self.decode_value(self.pc + 2)?;
                Op::Jf(cond, dest)
            }
            opcodes::OPCODE_OUT => {
                let arg_value = self.decode_value(self.pc + 1)?;
                Op::Out(arg_value)
            }
            opcodes::OPCODE_IN => {
                let dest = self.decode_value(self.pc + 1)?;
                Op::In(dest)
            }
            opcodes::OPCODE_NOOP => Op::Noop,

            opcodes::OPCODE_ADD => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Add(d, l, r)
            }
            opcodes::OPCODE_MULT => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Mult(d, l, r)
            }
            opcodes::OPCODE_MOD => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Mod(d, l, r)
            }
            opcodes::OPCODE_AND => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::And(d, l, r)
            }
            opcodes::OPCODE_OR => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Or(d, l, r)
            }
            opcodes::OPCODE_NOT => {
                let d = self.decode_value(self.pc + 1)?;
                let v = self.decode_value(self.pc + 2)?;
                Op::Not(d, v)
            }
            opcodes::OPCODE_RMEM => {
                let d = self.decode_value(self.pc + 1)?;
                let addr = self.decode_value(self.pc + 2)?;
                Op::Rmem(d, addr)
            }
            opcodes::OPCODE_WMEM => {
                let addr = self.decode_value(self.pc + 1)?;
                let value = self.decode_value(self.pc + 2)?;
                Op::Wmem(addr, value)
            }
            opcodes::OPCODE_CALL => {
                let dest = self.decode_value(self.pc + 1)?;
                Op::Call(dest)
            }
            opcodes::OPCODE_RET => Op::Ret,

            _ => {
                return Err(VMError::InvalidOpcode(value));
            }
        };
        Ok(op)
    }

    fn read(&self, value: Value) -> u16 {
        match value {
            Value::Literal(l) => l,
            Value::Register(reg) => self.registers[reg as usize],
        }
    }

    fn write_register(&mut self, target: Value, value: u16) -> Result<(), VMError> {
        match target {
            Value::Register(reg) => match self.registers.get_mut(reg as usize) {
                Some(slot) => {
                    *slot = value;
                    Ok(())
                }
                None => Err(VMError::InvalidRegister(Value::Register(reg))),
            },
            _ => Err(VMError::InvalidRegister(target)),
        }
    }

    fn read_address(&self, value: Value) -> Result<usize, VMError> {
        let addr = self.read(value) as usize;
        if addr >= MEMORY_SIZE {
            return Err(VMError::InvalidMemoryAddress(addr));
        }
        Ok(addr)
    }

    fn read_input_char(&mut self) -> Result<u16, VMError> {
        if let Some(value) = self.input_buffer.pop_front() {
            return Ok(value);
        }

        let mut line = String::new();
        let bytes_read = io::stdin()
            .read_line(&mut line)
            .map_err(|err| VMError::IoError(err.to_string()))?;
        if bytes_read == 0 {
            return Err(VMError::IoError("EOF while reading input".to_string()));
        }
        for byte in line.as_bytes() {
            self.input_buffer.push_back(*byte as u16);
        }

        self.input_buffer
            .pop_front()
            .ok_or_else(|| VMError::IoError("Empty input buffer".to_string()))
    }

    pub fn run(&mut self) -> Result<(), VMError> {
        let mut stdout = io::stdout();
        loop {
            let op = self.decode_op()?;
            let mut next_pc = self.pc + op.size();
            match op {
                Op::Halt => {
                    break;
                }
                Op::Set(reg, value) => {
                    let v = self.read(value);
                    self.write_register(reg, v)?;
                }
                Op::Push(value) => {
                    self.stack.push(self.read(value));
                }
                Op::Pop(value) => {
                    let Some(stack_value) = self.stack.pop() else {
                        return Err(VMError::EmptyStack);
                    };
                    self.write_register(value, stack_value)?;
                }
                Op::Eq(d, l, r) => {
                    let res = if self.read(l) == self.read(r) { 1 } else { 0 };
                    self.write_register(d, res)?;
                }
                Op::Gt(d, l, r) => {
                    let res = if self.read(l) > self.read(r) { 1 } else { 0 };
                    self.write_register(d, res)?;
                }
                Op::Jmp(dest) => {
                    next_pc = self.read_address(dest)?;
                }
                Op::Jt(cond, dest) => {
                    if self.read(cond) != 0 {
                        next_pc = self.read_address(dest)?;
                    }
                }
                Op::Jf(cond, dest) => {
                    if self.read(cond) == 0 {
                        next_pc = self.read_address(dest)?;
                    }
                }
                Op::Add(d, l, r) => {
                    let a = self.read(l);
                    let b = self.read(r);
                    let c = (a + b) % 32768;
                    self.write_register(d, c)?;
                }
                Op::Mult(d, l, r) => {
                    let a = self.read(l) as u32;
                    let b = self.read(r) as u32;
                    let c = ((a * b) % 32768) as u16;
                    self.write_register(d, c)?;
                }
                Op::Mod(d, l, r) => {
                    let b = self.read(r);
                    if b == 0 {
                        return Err(VMError::DivisionByZero);
                    }
                    let c = self.read(l) % b;
                    self.write_register(d, c)?;
                }
                Op::And(d, l, r) => {
                    let c = self.read(l) & self.read(r);
                    self.write_register(d, c)?;
                }
                Op::Or(d, l, r) => {
                    let c = self.read(l) | self.read(r);
                    self.write_register(d, c)?;
                }
                Op::Not(d, value) => {
                    let c = !self.read(value) & 0x7FFF;
                    self.write_register(d, c)?;
                }
                Op::Rmem(d, addr) => {
                    let address = self.read_address(addr)?;
                    let value = self.memory[address];
                    self.write_register(d, value)?;
                }
                Op::Wmem(addr, value) => {
                    let address = self.read_address(addr)?;
                    self.memory[address] = self.read(value);
                }
                Op::Call(dest) => {
                    self.stack.push(next_pc as u16);
                    next_pc = self.read_address(dest)?;
                }
                Op::Ret => {
                    let Some(address) = self.stack.pop() else {
                        break;
                    };
                    let address = address as usize;
                    if address >= MEMORY_SIZE {
                        return Err(VMError::InvalidMemoryAddress(address));
                    }
                    next_pc = address;
                }
                Op::Out(value) => {
                    let byte = self.read(value) as u8;
                    stdout
                        .write_all(&[byte])
                        .map_err(|err| VMError::IoError(err.to_string()))?;
                    stdout
                        .flush()
                        .map_err(|err| VMError::IoError(err.to_string()))?;
                }
                Op::In(dest) => {
                    let value = self.read_input_char()?;
                    self.write_register(dest, value)?;
                }
                Op::Noop => {}
            };
            self.pc = next_pc;
        }
        Ok(())
    }
}

mod opcodes {
    pub const OPCODE_HALT: u16 = 0;
    pub const OPCODE_SET: u16 = 1;
    pub const OPCODE_PUSH: u16 = 2;
    pub const OPCODE_POP: u16 = 3;
    pub const OPCODE_EQ: u16 = 4;
    pub const OPCODE_GT: u16 = 5;
    pub const OPCODE_JMP: u16 = 6;
    pub const OPCODE_JT: u16 = 7;
    pub const OPCODE_JF: u16 = 8;
    pub const OPCODE_ADD: u16 = 9;
    pub const OPCODE_MULT: u16 = 10;
    pub const OPCODE_MOD: u16 = 11;
    pub const OPCODE_AND: u16 = 12;
    pub const OPCODE_OR: u16 = 13;
    pub const OPCODE_NOT: u16 = 14;
    pub const OPCODE_RMEM: u16 = 15;
    pub const OPCODE_WMEM: u16 = 16;
    pub const OPCODE_CALL: u16 = 17;
    pub const OPCODE_RET: u16 = 18;
    pub const OPCODE_OUT: u16 = 19;
    pub const OPCODE_IN: u16 = 20;
    pub const OPCODE_NOOP: u16 = 21;
}

#[derive(Debug, PartialEq)]
pub enum Op {
    /// Opcode 0
    Halt,
    /// Opcode 1
    Set(Value, Value),
    /// Opcode 2
    Push(Value),
    /// Opcode 3
    Pop(Value),
    /// Opcode 4
    Eq(Value, Value, Value),
    /// Opcode 5
    Gt(Value, Value, Value),
    /// Opcode 6
    Jmp(Value),
    /// Opcode 7
    Jt(Value, Value),
    /// Opcode 8
    Jf(Value, Value),
    /// Opcode 9
    Add(Value, Value, Value),
    /// Opcode 10
    Mult(Value, Value, Value),
    /// Opcode 11
    Mod(Value, Value, Value),
    /// Opcode 12
    And(Value, Value, Value),
    /// Opcode 13
    Or(Value, Value, Value),
    /// Opcode 14
    Not(Value, Value),
    /// Opcode 15
    Rmem(Value, Value),
    /// Opcode 16
    Wmem(Value, Value),
    /// Opcode 17
    Call(Value),
    /// Opcode 18
    Ret,
    /// Opcode 19
    Out(Value),
    /// Opcode 20
    In(Value),
    /// Opcode 21
    Noop,
}

impl Op {
    fn size(&self) -> usize {
        match self {
            Op::Halt => 1,
            Op::Set(_, _) => 3,
            Op::Push(_) => 2,
            Op::Pop(_) => 2,
            Op::Eq(_, _, _) => 4,
            Op::Gt(_, _, _) => 4,
            Op::Jmp(_) => 2,
            Op::Jt(_, _) => 3,
            Op::Jf(_, _) => 3,
            Op::Add(_, _, _) => 4,
            Op::Mult(_, _, _) => 4,
            Op::Mod(_, _, _) => 4,
            Op::And(_, _, _) => 4,
            Op::Or(_, _, _) => 4,
            Op::Not(_, _) => 3,
            Op::Rmem(_, _) => 3,
            Op::Wmem(_, _) => 3,
            Op::Call(_) => 2,
            Op::Ret => 1,
            Op::Out(_) => 2,
            Op::In(_) => 2,
            Op::Noop => 1,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Literal(lit) => write!(f, "Literal {}", lit),
            Value::Register(reg) => write!(f, "Register #{}", reg),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vm::{Op, VM, VMError, Value, opcodes};

    fn encode_ops(ops: &[Op]) -> Vec<u16> {
        let mut v = Vec::new();
        for op in ops {
            match op {
                Op::Halt => v.push(opcodes::OPCODE_HALT),
                Op::Set(dst, value) => {
                    v.push(opcodes::OPCODE_SET);
                    v.push(dst.into());
                    v.push(value.into());
                }
                Op::Push(value) => {
                    v.push(opcodes::OPCODE_PUSH);
                    v.push(value.into());
                }
                Op::Pop(value) => {
                    v.push(opcodes::OPCODE_POP);
                    v.push(value.into());
                }
                Op::Eq(d, l, r) => {
                    v.push(opcodes::OPCODE_EQ);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Gt(d, l, r) => {
                    v.push(opcodes::OPCODE_GT);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Jmp(dest) => {
                    v.push(opcodes::OPCODE_JMP);
                    v.push(dest.into());
                }
                Op::Jt(cond, dest) => {
                    v.push(opcodes::OPCODE_JT);
                    v.push(cond.into());
                    v.push(dest.into());
                }
                Op::Jf(cond, dest) => {
                    v.push(opcodes::OPCODE_JF);
                    v.push(cond.into());
                    v.push(dest.into());
                }
                Op::Add(d, l, r) => {
                    v.push(opcodes::OPCODE_ADD);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Mult(d, l, r) => {
                    v.push(opcodes::OPCODE_MULT);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Mod(d, l, r) => {
                    v.push(opcodes::OPCODE_MOD);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::And(d, l, r) => {
                    v.push(opcodes::OPCODE_AND);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Or(d, l, r) => {
                    v.push(opcodes::OPCODE_OR);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Not(d, value) => {
                    v.push(opcodes::OPCODE_NOT);
                    v.push(d.into());
                    v.push(value.into());
                }
                Op::Rmem(d, addr) => {
                    v.push(opcodes::OPCODE_RMEM);
                    v.push(d.into());
                    v.push(addr.into());
                }
                Op::Wmem(addr, value) => {
                    v.push(opcodes::OPCODE_WMEM);
                    v.push(addr.into());
                    v.push(value.into());
                }
                Op::Call(dest) => {
                    v.push(opcodes::OPCODE_CALL);
                    v.push(dest.into());
                }
                Op::Ret => v.push(opcodes::OPCODE_RET),
                Op::Out(value) => {
                    v.push(opcodes::OPCODE_OUT);
                    v.push(value.into());
                }
                Op::In(value) => {
                    v.push(opcodes::OPCODE_IN);
                    v.push(value.into());
                }
                Op::Noop => v.push(opcodes::OPCODE_NOOP),
            }
        }
        v
    }

    #[test]
    fn test_encode_ops() {
        let ops = [Op::Push(Value::Literal(123)), Op::Pop(Value::Register(2))];
        let mem = encode_ops(&ops);
        let mut vm = VM::new_with_memory_slice(&mem);
        vm.run().expect("vm run ok");
        assert_eq!(vm.registers[2], 123);
    }

    #[test]
    fn test_value_types() {
        assert_eq!(Value::try_from(42), Ok(Value::Literal(42)));
        assert_eq!(Value::try_from(32770), Ok(Value::Register(2)));
        assert!(matches!(
            Value::try_from(u16::MAX),
            Err(VMError::InvalidValueError(_))
        ));
    }

    #[test]
    fn test_decode_halt_op_from_mem() {
        let mem = [opcodes::OPCODE_HALT];
        let mut vm = VM::new_with_memory_slice(&mem);
        let next_op = vm.decode_op();
        assert!(matches!(next_op, Ok(Op::Halt)));
    }

    #[test]
    fn test_decode_op_with_argument_from_mem() {
        let arg_data: u16 = 42;
        let mem = [opcodes::OPCODE_OUT, arg_data];
        let mut vm = VM::new_with_memory_slice(&mem);
        let next_op = vm.decode_op();
        assert_eq!(next_op, Ok(Op::Out(Value::Literal(arg_data))));
    }

    #[test]
    fn test_decode_multi_arg_op() {
        let mem = [9, 32768 + 1, 2, 32768 + 3];
        let mut vm = VM::new_with_memory_slice(&mem);
        let next_op = vm.decode_op();
        assert_eq!(
            next_op,
            Ok(Op::Add(
                Value::Register(1),
                Value::Literal(2),
                Value::Register(3)
            ))
        );
    }

    #[test]
    fn test_pc_state_after_op() {
        let mem = [19, 42];
        let mut vm = VM::new_with_memory_slice(&mem);
        let _ = vm.run();
        assert_eq!(vm.pc, 2);
    }

    #[test]
    fn test_decode_push_op() {
        let mem = [opcodes::OPCODE_PUSH, 32768 + 4];
        let mut vm = VM::new_with_memory_slice(&mem);
        let op = vm.decode_op();
        assert_eq!(op, Ok(Op::Push(Value::Register(4))));

        let mem = [opcodes::OPCODE_PUSH, 4];
        let mut vm = VM::new_with_memory_slice(&mem);
        let op = vm.decode_op();
        assert_eq!(op, Ok(Op::Push(Value::Literal(4))));
    }

    #[test]
    fn test_push_pop() {
        let mem = [opcodes::OPCODE_PUSH, 4, opcodes::OPCODE_POP, 3 + 32768];
        let mut vm = VM::new_with_memory_slice(&mem);
        vm.run().expect("VM run OK");
        assert_eq!(vm.registers[3], 4);
    }

    #[test]
    fn test_set_op() {
        let ops = [opcodes::OPCODE_SET, 32768 + 1, 500];
        let mut vm = VM::new_with_memory_slice(&ops);
        vm.run().unwrap();
        assert_eq!(vm.registers[1], 500);
    }

    #[test]
    fn test_add_op() {
        let ops = [
            opcodes::OPCODE_SET,
            32768 + 1,
            10,
            opcodes::OPCODE_SET,
            32768 + 2,
            20,
            opcodes::OPCODE_ADD,
            32768 + 3,
            32768 + 1,
            32768 + 2,
        ];
        let mut vm = VM::new_with_memory_slice(&ops);
        vm.run().unwrap();
        assert_eq!(vm.registers[3], 10 + 20);
    }
}
