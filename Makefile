EXE:=sph sph.gui omp-sph cuda-sph
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS=-lm

.PHONY: clean

sph: sph.c

omp: omp-sph

cuda: cuda-sph

gui: sph.gui

all: $(EXE)

cuda-sph: cuda-sph.cu
	nvcc $< $(LDLIBS) -o $@
omp-sph: CFLAGS+=-fopenmp
omp-sph: omp-sph.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@
sph.gui: CFLAGS+=-DGUI
sph.gui: LDLIBS+=-lglut -lGL -lX11
sph.gui: sph.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	\rm -f $(EXE) *.o *~
