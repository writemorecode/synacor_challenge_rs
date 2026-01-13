use std::fmt::Display;

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
}

impl Default for VM {
    fn default() -> Self {
        Self {
            registers: [0u16; REGISTER_COUNT],
            memory: [0u16; MEMORY_SIZE],
            stack: Vec::new(),
            pc: 0,
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

    fn decode_op(&mut self) -> Result<Op, VMError> {
        let value = self.read_value_at_pc()?;
        let Value::Literal(lit_val) = value else {
            return Err(VMError::InvalidOpcode(value));
        };
        let op = match lit_val {
            opcodes::OPCODE_HALT => Op::Halt,
            opcodes::OPCODE_OUT => {
                let arg_value = self.decode_value(self.pc + 1)?;
                Op::Out(arg_value)
            }
            opcodes::OPCODE_NOOP => Op::Noop,

            opcodes::OPCODE_PUSH => {
                let arg_value = self.decode_value(self.pc + 1)?;
                Op::Push(arg_value)
            }
            opcodes::OPCODE_POP => {
                let arg_value = self.decode_value(self.pc + 1)?;
                let Value::Register(_) = arg_value else {
                    return Err(VMError::InvalidRegister(arg_value));
                };
                Op::Pop(arg_value)
            }

            opcodes::OPCODE_ADD => {
                let d = self.decode_value(self.pc + 1)?;
                let l = self.decode_value(self.pc + 2)?;
                let r = self.decode_value(self.pc + 3)?;
                Op::Add(d, l, r)
            }

            opcodes::OPCODE_SET => {
                let reg = self.decode_value(self.pc + 1)?;
                let val = self.decode_value(self.pc + 2)?;
                Op::Set(reg, val)
            }

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

    pub fn run(&mut self) -> Result<(), VMError> {
        loop {
            let op = self.decode_op()?;
            dbg!(&op);
            match op {
                Op::Halt => {
                    println!("HALTING");
                    break;
                }
                Op::Out(ref value) => {
                    println!("OUTPUT: '{}'", value);
                }
                Op::Noop => {
                    println!("NOOP");
                }
                Op::Push(value) => {
                    println!("Pushing value {}", value);
                    match value {
                        Value::Literal(l) => self.stack.push(l),
                        Value::Register(reg) => self.stack.push(self.registers[reg as usize]),
                    };
                }
                Op::Pop(value) => {
                    println!("Popping into register {}", value);
                    let Some(stack_value) = self.stack.pop() else {
                        return Err(VMError::EmptyStack);
                    };
                    let Value::Register(reg) = value else {
                        return Err(VMError::InvalidRegister(value));
                    };
                    self.registers[reg as usize] = stack_value;
                }
                Op::Add(d, l, r) => {
                    let Value::Register(reg) = d else {
                        return Err(VMError::InvalidRegister(d));
                    };
                    let a = self.read(l);
                    let b = self.read(r);
                    let c = (a + b) % 32768;
                    self.registers[reg as usize] = c;
                }
                Op::Set(reg, value) => {
                    let Value::Register(r) = reg else {
                        return Err(VMError::InvalidRegister(reg));
                    };
                    let v = self.read(value);
                    self.registers[r as usize] = v;
                }
            };
            self.pc += op.size();
        }
        Ok(())
    }
}

mod opcodes {
    pub const OPCODE_HALT: u16 = 0;
    pub const OPCODE_SET: u16 = 1;
    pub const OPCODE_OUT: u16 = 19;
    pub const OPCODE_NOOP: u16 = 21;

    pub const OPCODE_PUSH: u16 = 2;
    pub const OPCODE_POP: u16 = 3;

    pub const OPCODE_ADD: u16 = 9;
}

#[derive(Debug, PartialEq)]
pub enum Op {
    /// Opcode 0
    Halt,
    /// Opcode 1
    Set(Value, Value),
    /// Opcode 19
    Out(Value),
    /// Opcode 21
    Noop,
    /// Opcode 2
    Push(Value),
    /// Opcode 3
    Pop(Value),
    /// Opcode 9
    Add(Value, Value, Value),
}

impl Op {
    fn size(&self) -> usize {
        match self {
            Op::Halt => 1,
            Op::Set(_, _) => 3,
            Op::Out(_) => 2,
            Op::Noop => 1,
            Op::Push(_) => 2,
            Op::Pop(_) => 2,
            Op::Add(_, _, _) => 4,
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
    use crate::vm::{
        Op, VM, VMError, Value,
        opcodes::{
            self, OPCODE_ADD, OPCODE_HALT, OPCODE_NOOP, OPCODE_OUT, OPCODE_POP, OPCODE_PUSH,
            OPCODE_SET,
        },
    };

    fn encode_ops(ops: &[Op]) -> Vec<u16> {
        let mut v = Vec::new();
        for op in ops {
            match op {
                Op::Halt => v.push(OPCODE_HALT),
                Op::Out(value) => {
                    v.push(OPCODE_OUT);
                    v.push(value.into());
                }
                Op::Noop => v.push(OPCODE_NOOP),
                Op::Push(value) => {
                    v.push(OPCODE_PUSH);
                    v.push(value.into());
                }
                Op::Pop(value) => {
                    v.push(OPCODE_POP);
                    v.push(value.into());
                }
                Op::Add(d, l, r) => {
                    v.push(OPCODE_ADD);
                    v.push(d.into());
                    v.push(l.into());
                    v.push(r.into());
                }
                Op::Set(dst, value) => {
                    v.push(OPCODE_SET);
                    v.push(dst.into());
                    v.push(value.into());
                }
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
        let mem = [OPCODE_PUSH, 32768 + 4];
        let mut vm = VM::new_with_memory_slice(&mem);
        let op = vm.decode_op();
        assert_eq!(op, Ok(Op::Push(Value::Register(4))));

        let mem = [OPCODE_PUSH, 4];
        let mut vm = VM::new_with_memory_slice(&mem);
        let op = vm.decode_op();
        assert_eq!(op, Ok(Op::Push(Value::Literal(4))));
    }

    #[test]
    fn test_push_pop() {
        let mem = [OPCODE_PUSH, 4, OPCODE_POP, 3 + 32768];
        let mut vm = VM::new_with_memory_slice(&mem);
        vm.run().expect("VM run OK");
        assert_eq!(vm.registers[3], 4);
    }

    #[test]
    fn test_set_op() {
        let ops = [OPCODE_SET, 32768 + 1, 500];
        let mut vm = VM::new_with_memory_slice(&ops);
        vm.run().unwrap();
        assert_eq!(vm.registers[1], 500);
    }

    #[test]
    fn test_add_op() {
        let ops = [
            OPCODE_SET,
            32768 + 1,
            10,
            OPCODE_SET,
            32768 + 2,
            20,
            OPCODE_ADD,
            32768 + 3,
            32768 + 1,
            32768 + 2,
        ];
        let mut vm = VM::new_with_memory_slice(&ops);
        vm.run().unwrap();
        assert_eq!(vm.registers[3], 10 + 20);
    }
}
