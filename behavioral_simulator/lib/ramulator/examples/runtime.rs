use std::mem::ManuallyDrop;

#[tokio::main]
async fn main() {
    let new_1c = || ManuallyDrop::new(ramulator::Ramulator::ddr4_preset(1).unwrap());

    let new_2c = || ManuallyDrop::new(ramulator::Ramulator::ddr4_preset(2).unwrap());

    println!("DDR4 Single-Channel Sequential");
    memory::testutils::sequential_1m(new_1c()).await;

    println!("DDR4 Dual-Channel Sequential");
    memory::testutils::sequential_1m(new_2c()).await;

    println!("DDR4 Single-Channel Random");
    memory::testutils::random_1m(new_1c()).await;

    println!("DDR4 Dual-Channel Random");
    memory::testutils::random_1m(new_2c()).await;
}
