use std::time::Instant;

struct ChargeSession {
    start_time: Instant,
    end_time: Instant,
    day_of_week: u8,
    start_soc: f32,
    end_soc: f32,
}
