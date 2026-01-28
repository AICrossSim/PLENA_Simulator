use memory::SimpleTiming;

#[tokio::main]
async fn main() {
    let new_1c = || {
        let simple = SimpleTiming::preset_ddr4_2400p(1);
        simple
    };

    let new_2c = || {
        let simple = SimpleTiming::preset_ddr4_2400p(2);
        simple
    };

    println!("DDR4 Single-Channel Sequential");
    memory::testutils::sequential_1m(new_1c()).await;

    println!("DDR4 Dual-Channel Sequential");
    memory::testutils::sequential_1m(new_2c()).await;

    println!("DDR4 Single-Channel Random");
    memory::testutils::random_1m(new_1c()).await;

    println!("DDR4 Dual-Channel Random");
    memory::testutils::random_1m(new_2c()).await;
}
