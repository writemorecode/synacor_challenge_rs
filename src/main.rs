use vm::vm::VM;

fn main() {
    let mem = [19, 42, 21, 19, 100];
    let mut vm = VM::new_with_memory_slice(&mem);
    let _ = vm.run();
}
