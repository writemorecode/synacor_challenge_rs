use vm::vm::{Value};

fn main() {
    println!("hello world!");
    let val = Value::Literal(42);
    dbg!(&val);
}
