# Shopping Sessions (HCMUT TMT): Rust pipeline, std only (no crates).
# Built natively on the linux/amd64 scorer; x86-64-v3 = AVX2/BMI2 (every Cascade Lake+ Xeon).
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY rust/ /src/
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release --locked

FROM debian:bookworm-slim
# The scorer runs the container as uid 1000; give that uid a name and a home.
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/sessions /app/sessions
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh /app/sessions
WORKDIR /app
CMD ["/app/run.sh"]
