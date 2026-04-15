FROM rust:1.91.1-slim

WORKDIR /app
COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN cargo install --path .

ENV XDG_CONFIG_HOME=/var/lib \
    XDG_DATA_HOME=/var/lib

VOLUME ["/var/lib/rycharger"]

CMD ["rycharger"]
