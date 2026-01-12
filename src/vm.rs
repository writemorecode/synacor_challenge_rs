#[derive(Debug, PartialEq)]
pub enum Value {
    Literal(u16),
    Register(u16),
}

#[derive(Debug, PartialEq)]
pub struct InvalidValueError(u16);

impl TryFrom<u16> for Value {
    type Error = InvalidValueError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0_u16..=32767 => Ok(Value::Literal(value)),
            32768_u16..=32775 => Ok(Value::Register(value - 32768)),
            32776..=u16::MAX => Err(InvalidValueError(value)),
        }
    }
}

const MEMORY_SIZE: usize = 2 << 16 - 1;
const REGISTER_COUNT: usize = 8;

pub struct VM {
    registers: [u16; REGISTER_COUNT],
    memory: [u16; MEMORY_SIZE],
}

impl VM {
    pub fn new() -> Self {
        Self {
            registers: [0u16; REGISTER_COUNT],
            memory: [0u16; MEMORY_SIZE],
        }
    }

    fn read(&self, value: Value) -> u16 {
        match value {
            Value::Literal(addr) => self.memory[addr as usize],
            Value::Register(reg) => self.registers[reg as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vm::{InvalidValueError, Value};

    #[test]
    fn test_value_types() {
        assert_eq!(Value::try_from(42), Ok(Value::Literal(42)));
        assert_eq!(Value::try_from(32770), Ok(Value::Register(2)));
        assert!(matches!(
            Value::try_from(u16::MAX),
            Err(InvalidValueError(_))
        ));
    }
}
