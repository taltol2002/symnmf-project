CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

# The final executable name must be symnmf
symnmf: symnmf.o
	$(CC) $(CFLAGS) symnmf.o -o symnmf $(LDFLAGS)

# Compilation of the source file into an object file
symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

# Clean rule
clean:
	rm -f *.o symnmf