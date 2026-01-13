use std::{fs, io};

fn main() -> io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "challenge.bin".to_string());
    let bytes = fs::read(&path)?;
    let mut vm = vm::vm::VM::new_with_program_bytes(&bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("{err:?}")))?;
    vm.run()
        .map_err(|err| io::Error::other(format!("{err:?}")))?;
    Ok(())
}
