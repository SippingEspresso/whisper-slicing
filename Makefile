BINARY_NAME=whisper-slicing
PREFIX?=$(HOME)/.local/bin

.PHONY: all build install clean test

all: build

build:
	go build -ldflags="-s -w" -o $(BINARY_NAME) .

install: build
	mkdir -p $(PREFIX)
	cp $(BINARY_NAME) $(PREFIX)/$(BINARY_NAME)
	ln -sf $(PREFIX)/$(BINARY_NAME) $(PREFIX)/whisper-parallel

clean:
	rm -f $(BINARY_NAME) whisper-parallel

test:
	go vet ./...
