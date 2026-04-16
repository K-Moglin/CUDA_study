PYTHON ?= python
CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra

.PHONY: print-config gen-mlp gen-router gen-route gen-naive-moe gen-moe gen-all build test-mlp test-router test-all

print-config:
	$(PYTHON) scripts/print_deepseek_v3_moe_config.py

gen-mlp:
	$(PYTHON) generate_tests/generate_mlp_cases.py

gen-router:
	$(PYTHON) generate_tests/generate_router_cases.py

gen-route:
	$(PYTHON) generate_tests/generate_route_cases.py

gen-naive-moe:
	$(PYTHON) generate_tests/generate_naive_moe_cases.py

gen-moe:
	$(PYTHON) generate_tests/generate_moe_cases.py

gen-all: gen-mlp gen-router gen-route gen-naive-moe gen-moe

build:
	$(CC) $(CFLAGS) -o c_impl/test_moe.exe c_impl/moe.c c_impl/test_moe.c -lm

test-mlp: gen-mlp
	$(CC) $(CFLAGS) -o c_impl/test_moe.exe c_impl/moe.c c_impl/test_moe.c -lm
	cd c_impl && .\test_moe.exe

test-router: gen-mlp gen-router
	$(CC) $(CFLAGS) -o c_impl/test_moe.exe c_impl/moe.c c_impl/test_moe.c -lm
	cd c_impl && .\test_moe.exe

test-all: gen-all build
	cd c_impl && .\test_moe.exe
